"""Host-side semantic analysis: the 18 typed relations + KL features + closure regions.

This module is the host twin of the ast_shannon extractor (shannon_pipeline.py).
Deriving relations on the *host* with the SAME detectors, the SAME 18-relation
vocabulary, the SAME closure operator, and the SAME feature mapping puts host
regions and historical patterns into one shared coordinate system:

    extraction (bug corpus) : locate sink -> closure -> DIE -> KL weights
    matching  (host seed)   : propose anchor -> closure region -> feasibility -> evidence

Every match component therefore inverts one extraction phase instead of being
an ad-hoc heuristic. Spans here are byte offsets from the javac worker (always
complete), whereas the historical ASTs used UTF-16 offsets; the detectors only
perform ordering/containment comparisons, which are offset-encoding agnostic.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
import math
import re
from typing import Any, Iterable

from .parser import JavaSyntaxTree


# ---------------------------------------------------------------------------
# Kind groups (mirrors the extractor's constants)
# ---------------------------------------------------------------------------

STATEMENT_ANCHOR_KINDS = {
    'VARIABLE', 'METHOD', 'CLASS', 'INTERFACE', 'ENUM', 'RECORD', 'EXPRESSION_STATEMENT',
    'RETURN', 'IF', 'WHILE_LOOP', 'FOR_LOOP', 'ENHANCED_FOR_LOOP', 'DO_WHILE_LOOP',
    'SWITCH', 'TRY', 'THROW', 'LAMBDA_EXPRESSION', 'METHOD_INVOCATION',
    'ASSIGNMENT', 'TYPE_CAST', 'NEW_CLASS', 'ASSERT', 'ERRONEOUS',
    'CONDITIONAL_EXPRESSION', 'INSTANCE_OF', 'PARENTHESIZED',
}

SCOPE_KINDS = {'CLASS', 'INTERFACE', 'ENUM', 'RECORD', 'METHOD', 'LAMBDA_EXPRESSION',
               'BLOCK', 'FOR_LOOP', 'ENHANCED_FOR_LOOP', 'CATCH'}

# Statement/declaration-level kinds eligible as entropy-ranked sink candidates
# (mirrors the extractor's SINK_CANDIDATE_KINDS; used for anchor refinement).
SINK_CANDIDATE_SET = {
    'CLASS', 'INTERFACE', 'ENUM', 'RECORD', 'METHOD', 'VARIABLE',
    'EXPRESSION_STATEMENT', 'IF', 'WHILE_LOOP', 'FOR_LOOP', 'ENHANCED_FOR_LOOP',
    'DO_WHILE_LOOP', 'SWITCH', 'TRY', 'SYNCHRONIZED', 'RETURN', 'THROW',
    'LAMBDA_EXPRESSION', 'NEW_CLASS', 'METHOD_INVOCATION', 'ASSIGNMENT',
    'TYPE_CAST', 'CONDITIONAL_EXPRESSION', 'INSTANCE_OF', 'ASSERT', 'ERRONEOUS',
}

ENCLOSING_DECL_KINDS = {'CLASS', 'INTERFACE', 'ENUM', 'RECORD', 'METHOD'}

TYPE_DECL_KINDS = ('CLASS', 'INTERFACE', 'ENUM', 'RECORD')

CONTROL_KINDS = {'WHILE_LOOP', 'FOR_LOOP', 'ENHANCED_FOR_LOOP', 'DO_WHILE_LOOP',
                 'SWITCH', 'TRY', 'SYNCHRONIZED', 'LAMBDA_EXPRESSION'}

DEF_USE_CAP = 400          # host seeds are whole programs; allow more edges than per-bug cap
MAX_REGION_NODES = 24      # same budget as pattern extraction

RELATION_TO_FEATURE = {
    'ENCLOSING_INSTANCE': 'InnerClass_Enclosing_Binding',
    'LAMBDA_CAPTURE': 'Lambda_Capture',
    'LOCAL_CLASS_CAPTURE': 'LocalClass_Capture',
    'EARLY_REFERENCE': 'Early_Construction_Reference',
    'FORWARD_DEF_USE': 'ForwardReference',
    'VARIABLE_SHADOWING': 'Variable_Shadowing',
    'EXTENDS': 'Inheritance',
    'IMPLEMENTS': 'Interface_Implementation',
    'OVERRIDING': 'Polymorphic_Overriding',
    'SUPER_MEMBER_ACCESS': 'Super_Member_Access',
    'TYPE_PARAMETER_BINDING': 'Generic_Inference',
    'GENERIC_INSTANTIATION': 'Generic_Inference',
    'TYPE_CAST_RELATION': 'TypeCast_Relation',
    'RECORD_DECONSTRUCTION': 'Record_Deconstruction',
    'DEF_USE': 'Plain_DefUse',
    'FLOW_SCOPING': 'Flow_Scoping',
    'SWITCH_EXHAUSTIVENESS': 'Switch_Exhaustiveness',
    'PATTERN_VARIABLE_FLOW': 'Pattern_Variable_Flow',
}


# ---------------------------------------------------------------------------
# Source-text heuristics (ported verbatim from the extractor)
# ---------------------------------------------------------------------------

MODIFIER_WORDS = {'public', 'private', 'protected', 'static', 'final', 'volatile',
                  'transient', 'abstract', 'native', 'synchronized', 'default',
                  'strictfp', 'sealed', 'non-sealed'}

_CLASS_HDR = re.compile(r'\b(class|interface|enum|record)\s+(\w+)')
_METHOD_NAME = re.compile(r'\b(\w+)\s*\(')


def parse_class_header(text):
    """(name, [extends], [implements], [permits]) from a type-declaration header window."""
    head = text.split('{', 1)[0]
    m = _CLASS_HDR.search(head)
    if not m:
        return None
    name = m.group(2)
    after = head[m.end():]

    def clause(word):
        cm = re.search(r'\b%s\s+([\w.]+(?:\s*,\s*[\w.]+)*)' % word, after)
        return [t.strip() for t in cm.group(1).split(',')] if cm else []

    return name, clause('extends'), clause('implements'), clause('permits')


def parse_method_name(text):
    m = _METHOD_NAME.search(text[:200])
    return m.group(1) if m else None


def parse_var_decl(text):
    """(name, declared_type) from variable/parameter text (no-terminator params, arrays, generics)."""
    t = text.strip()
    while True:
        m = re.match(r'^(?:public|private|protected|static|final|volatile|transient)\s+', t)
        if m:
            t = t[m.end():]
            continue
        m = re.match(r'^@\w+(?:\([^)]*\))?\s*', t)
        if m:
            t = t[m.end():]
            continue
        break
    body = re.split(r'[=;:\n]', t, 1)[0].strip().rstrip(',)').strip()
    if not body:
        return None, None
    protected = re.sub(r'<[^<>]*>', '<>', body)
    toks = protected.split()
    if not toks:
        return None, None
    last = toks[-1]
    arr = ''
    m = re.match(r'^(\w+)((?:\[\])+)$', last)
    if m:
        last, arr = m.group(1), '[]' * (len(m.group(2)) // 2)
    if not re.match(r'^\w+$', last) or last in MODIFIER_WORDS:
        return None, None
    tail = body.rstrip()
    typ = tail[:len(tail) - len(toks[-1])].strip()
    if arr:
        typ = (typ + arr).strip()
    if not typ:
        return last, '(inferred)'
    return last, re.sub(r'\s+', ' ', typ)


def member_select_name(text):
    t = text.strip().rstrip(')').split('.')[-1].strip()
    m = re.match(r'^(\w+)', t)
    return m.group(1) if m else None


def member_select_qualifier(text):
    if re.search(r'(?:^|\.)super\s*\.', text.strip()):
        return 'super'
    return None


def param_type_base(text):
    return text.strip().split('<', 1)[0].strip()


# ---------------------------------------------------------------------------
# HostGraph: semantic layer over one parsed host tree
# ---------------------------------------------------------------------------

class HostGraph:
    """Statement-anchored relation graph over a javac-parsed host program."""

    def __init__(self, tree: JavaSyntaxTree) -> None:
        self.tree = tree
        self._text: dict[int, str] = {}
        self._relations: list[dict[str, Any]] | None = None
        self._edge_index: dict[int, list[dict[str, Any]]] | None = None

    # -- structure helpers ----------------------------------------------------

    def node_text(self, nid: int) -> str:
        text = self._text.get(nid)
        if text is None:
            node = self.tree.nodes[nid]
            text = self.tree.snippet(node.start_byte, node.end_byte)
            self._text[nid] = text
        return text

    def scope_chain(self, nid: int) -> list[int]:
        chain = []
        cur = self.tree.nodes[nid].parent_id
        while cur is not None:
            if self.tree.nodes[cur].raw_kind in SCOPE_KINDS:
                chain.append(cur)
            cur = self.tree.nodes[cur].parent_id
        return chain

    def nearest_ancestor(self, nid: int, kinds: set[str]) -> int | None:
        cur = self.tree.nodes[nid].parent_id
        while cur is not None:
            if self.tree.nodes[cur].raw_kind in kinds:
                return cur
            cur = self.tree.nodes[cur].parent_id
        return None

    def nearest_enclosing_decl(self, nid: int) -> int | None:
        a = self.nearest_ancestor(nid, ENCLOSING_DECL_KINDS)
        return a if a is not None else self.tree.root_id

    def anchor_statement(self, nid: int) -> int | None:
        cur = nid
        while cur is not None:
            if self.tree.nodes[cur].raw_kind in STATEMENT_ANCHOR_KINDS:
                return cur
            cur = self.tree.nodes[cur].parent_id
        return self.tree.root_id

    def is_inside(self, nid: int, ancestor_id: int) -> bool:
        cur = self.tree.nodes[nid].parent_id
        while cur is not None:
            if cur == ancestor_id:
                return True
            cur = self.tree.nodes[cur].parent_id
        return False

    # -- symbol table ----------------------------------------------------------

    def build_symbol_table(self):
        decls = defaultdict(list)
        by_name = defaultdict(list)
        for nid, node in sorted(self.tree.nodes.items()):
            kind = node.raw_kind
            if kind == 'VARIABLE':
                name, typ = parse_var_decl(self.node_text(nid))
                if name:
                    scope = node.parent_id if node.parent_id is not None else self.tree.root_id
                    decls[scope].append((name, nid, 'var', typ))
                    by_name[name].append((scope, nid, 'var', typ))
            elif kind in TYPE_DECL_KINDS:
                hdr = parse_class_header(self.node_text(nid))
                if hdr:
                    scope = node.parent_id if node.parent_id is not None else self.tree.root_id
                    decls[scope].append((hdr[0], nid, 'type', hdr))
                    by_name[hdr[0]].append((scope, nid, 'type', hdr))
            elif kind == 'METHOD':
                name = parse_method_name(self.node_text(nid))
                if name:
                    scope = node.parent_id if node.parent_id is not None else self.tree.root_id
                    decls[scope].append((name, nid, 'method', None))
                    by_name[name].append((scope, nid, 'method', None))
            elif kind == 'TYPE_PARAMETER':
                name = self.node_text(nid).strip()
                if name and re.match(r'^\w+$', name):
                    scope = node.parent_id if node.parent_id is not None else self.tree.root_id
                    decls[scope].append((name, nid, 'tparam', None))
                    by_name[name].append((scope, nid, 'tparam', None))
        return decls, by_name

    def resolve(self, use_nid: int, name: str, by_name):
        for scope in [use_nid] + self.scope_chain(use_nid):
            for (sid, did, kind, typ) in by_name.get(name, ()):
                if sid == scope:
                    return did, kind, typ, sid
        return None, None, None, None

    # -- relation derivation (18 detectors, extractor-faithful) -----------------

    @property
    def relations(self) -> list[dict[str, Any]]:
        if self._relations is None:
            self._relations = self.derive_relations()
        return self._relations

    def edge_index(self) -> dict[int, list[dict[str, Any]]]:
        if self._edge_index is None:
            index: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for edge in self.relations:
                index[edge['source']].append(edge)
                index[edge['target']].append(edge)
            self._edge_index = index
        return self._edge_index

    def derive_relations(self) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        seen = set()

        def add(src, dst, typ):
            if src is None or dst is None or src == dst:
                return
            a, b = self.anchor_statement(src), self.anchor_statement(dst)
            if a is None or b is None or a == b:
                return
            key = (a, b, typ)
            if key not in seen:
                seen.add(key)
                edges.append({'source': a, 'target': b, 'type': typ})

        _decls, by_name = self.build_symbol_table()

        classes = {nid: node for nid, node in self.tree.nodes.items()
                   if node.raw_kind in TYPE_DECL_KINDS}
        class_hdr = {}
        class_name = {}
        for cid in classes:
            hdr = parse_class_header(self.node_text(cid))
            class_hdr[cid] = hdr
            if hdr:
                class_name[hdr[0]] = cid

        # (6) EXTENDS / (7) IMPLEMENTS -> real internal type nodes
        for cid, hdr in class_hdr.items():
            if not hdr:
                continue
            for ext in hdr[1]:
                tgt = class_name.get(ext.split('.')[-1])
                if tgt is not None:
                    add(cid, tgt, 'EXTENDS')
            for imp in hdr[2]:
                tgt = class_name.get(imp.split('.')[-1])
                if tgt is not None:
                    add(cid, tgt, 'IMPLEMENTS')

        # (8) OVERRIDING: same-name methods along internal inheritance chains
        def direct_methods(cls):
            out = {}
            for ch in self.tree.nodes[cls].child_ids:
                if self.tree.nodes[ch].raw_kind == 'METHOD':
                    nm = parse_method_name(self.node_text(ch))
                    if nm:
                        out[nm] = ch
            return out

        for cid, hdr in class_hdr.items():
            if not hdr or not hdr[1]:
                continue
            for ext in hdr[1]:
                parent = class_name.get(ext.split('.')[-1])
                if parent is None:
                    continue
                pm = direct_methods(parent)
                for nm, m in direct_methods(cid).items():
                    if nm in pm:
                        add(m, pm[nm], 'OVERRIDING')

        # (1) ENCLOSING_INSTANCE: inner class -> outer (records are implicitly static)
        for cid, c in classes.items():
            outer = self.nearest_ancestor(cid, set(TYPE_DECL_KINDS))
            if outer is not None and c.raw_kind == 'CLASS' and self.tree.nodes[outer].raw_kind != 'RECORD':
                add(cid, outer, 'ENCLOSING_INSTANCE')

        lambda_of = [nid for nid, n in self.tree.nodes.items() if n.raw_kind == 'LAMBDA_EXPRESSION']
        pattern_binding = {}
        def_use_count = 0
        for nid, n in sorted(self.tree.nodes.items()):
            txt = self.node_text(nid)
            if n.raw_kind == 'IDENTIFIER':
                name = txt.strip()
                if not name or not re.match(r'^\w+$', name):
                    continue
                did, kind, typ, _scope = self.resolve(nid, name, by_name)
                if did is None:
                    continue
                use_anchor = self.anchor_statement(nid)
                # (2) LAMBDA_CAPTURE
                in_lambda = next((l for l in lambda_of if self.is_inside(nid, l)), None)
                if in_lambda is not None and not self.is_inside(did, in_lambda):
                    add(did, use_anchor, 'LAMBDA_CAPTURE')
                # (3) LOCAL_CLASS_CAPTURE
                if kind == 'var':
                    cls = self.nearest_ancestor(nid, set(TYPE_DECL_KINDS))
                    if cls is not None:
                        mth = self.nearest_ancestor(cls, {'METHOD'})
                        if mth is not None and self.is_inside(did, mth) and not self.is_inside(did, cls):
                            decl_m = self.nearest_ancestor(did, {'METHOD'})
                            if decl_m == mth:
                                add(did, use_anchor, 'LOCAL_CLASS_CAPTURE')
                forward = self.tree.nodes[nid].start_byte < self.tree.nodes[did].start_byte
                if forward:
                    # (4) EARLY_REFERENCE / (15) FORWARD_DEF_USE
                    enclosing_var = self.nearest_ancestor(nid, {'VARIABLE'})
                    use_in_field_init = (kind == 'var' and enclosing_var is not None
                                         and self.nearest_ancestor(enclosing_var, {'CLASS'}) is not None
                                         and self.nearest_ancestor(did, {'CLASS'}) ==
                                             self.nearest_ancestor(enclosing_var, {'CLASS'}))
                    if use_in_field_init:
                        add(did, use_anchor, 'EARLY_REFERENCE')
                    else:
                        add(did, use_anchor, 'FORWARD_DEF_USE')
                else:
                    # (14) DEF_USE
                    if def_use_count < DEF_USE_CAP:
                        add(did, use_anchor, 'DEF_USE')
                        def_use_count += 1
                # (10) TYPE_PARAMETER_BINDING
                if kind == 'tparam':
                    add(did, use_anchor, 'TYPE_PARAMETER_BINDING')
            elif n.raw_kind == 'MEMBER_SELECT':
                # (9) SUPER_MEMBER_ACCESS
                if member_select_qualifier(txt) == 'super':
                    mname = member_select_name(txt)
                    cls = self.nearest_ancestor(nid, set(TYPE_DECL_KINDS))
                    target = None
                    if cls is not None and class_hdr.get(cls):
                        for ext in class_hdr[cls][1]:
                            parent = class_name.get(ext.split('.')[-1])
                            if parent is not None:
                                for ch in self.tree.nodes[parent].child_ids:
                                    if self.tree.nodes[ch].raw_kind == 'METHOD' and \
                                       parse_method_name(self.node_text(ch)) == mname:
                                        target = ch
                                        break
                            if target:
                                break
                    add(nid, target if target is not None else cls, 'SUPER_MEMBER_ACCESS')
            elif n.raw_kind == 'PARAMETERIZED_TYPE':
                # (11) GENERIC_INSTANTIATION
                base = param_type_base(txt)
                tgt = class_name.get(base.split('.')[-1])
                if tgt is not None:
                    add(nid, tgt, 'GENERIC_INSTANTIATION')
                for ch in self.tree.nodes[nid].child_ids:
                    if self.tree.nodes[ch].raw_kind == 'IDENTIFIER':
                        did, kind, _, _ = self.resolve(ch, self.node_text(ch).strip(), by_name)
                        if kind == 'tparam':
                            add(did, self.anchor_statement(nid), 'TYPE_PARAMETER_BINDING')
            elif n.raw_kind == 'TYPE_CAST':
                # (12) TYPE_CAST_RELATION
                m = re.match(r'^\(\s*([\w.]+)', txt.strip())
                if m:
                    tgt = class_name.get(m.group(1).split('.')[-1])
                    if tgt is not None:
                        add(nid, tgt, 'TYPE_CAST_RELATION')
            elif n.raw_kind in ('INSTANCE_OF', 'DECONSTRUCTION_PATTERN'):
                # (13) RECORD_DECONSTRUCTION + pattern binding bookkeeping
                if n.raw_kind == 'DECONSTRUCTION_PATTERN':
                    m2 = re.match(r'\s*(?:record\s+)?(\w+)\s*\(', txt)
                    if m2:
                        tgt = class_name.get(m2.group(1))
                        if tgt is not None:
                            add(nid, tgt, 'RECORD_DECONSTRUCTION')
                bind = re.search(r'(\w+)\s*\)\s*$', txt.strip()) or re.search(
                    r'(?:instanceof\s+[\w.<>\[\]]+\s+(\w+))', txt)
                if bind:
                    pattern_binding[nid] = bind.group(1)
            elif n.raw_kind == 'SWITCH':
                # (17) SWITCH_EXHAUSTIVENESS
                stack2 = list(self.tree.nodes[nid].child_ids)
                pat_label = None
                while stack2:
                    cur = stack2.pop()
                    if self.tree.nodes[cur].raw_kind == 'PATTERN_CASE_LABEL':
                        pat_label = cur
                        break
                    stack2.extend(self.tree.nodes[cur].child_ids)
                sel = next((c for c in self.tree.nodes[nid].child_ids
                            if self.tree.nodes[c].raw_kind in
                            ('IDENTIFIER', 'MEMBER_SELECT', 'PARENTHESIZED')), None)
                sel_type = None
                if sel is not None:
                    cur = sel
                    while self.tree.nodes[cur].raw_kind == 'PARENTHESIZED':
                        inner = next((c for c in self.tree.nodes[cur].child_ids
                                      if self.tree.nodes[c].raw_kind in
                                      ('IDENTIFIER', 'MEMBER_SELECT')), None)
                        if inner is None:
                            break
                        cur = inner
                    tname = self.node_text(cur).strip().split('.', 1)[0]
                    if not re.match(r'^\w+$', tname):
                        tname = member_select_name(tname) or ''
                    if re.match(r'^\w+$', tname):
                        did, kind, typ, _ = self.resolve(cur, tname, by_name)
                        if kind == 'type':
                            sel_type = tname
                        elif kind == 'var' and typ and typ != '(inferred)':
                            sel_type = param_type_base(typ)
                if sel_type:
                    tgt = class_name.get(sel_type.split('.')[-1])
                    if tgt is not None:
                        add(nid, tgt, 'SWITCH_EXHAUSTIVENESS')
                if pat_label is not None:
                    add(nid, pat_label, 'SWITCH_EXHAUSTIVENESS')

        # (18) PATTERN_VARIABLE_FLOW
        for pnid, bname in pattern_binding.items():
            for nid, n in self.tree.nodes.items():
                if n.raw_kind == 'IDENTIFIER' and self.node_text(nid).strip() == bname and nid != pnid:
                    if self.is_inside(nid, self.anchor_statement(pnid)):
                        if self.tree.nodes[nid].start_byte >= self.tree.nodes[pnid].start_byte:
                            add(pnid, self.anchor_statement(nid), 'PATTERN_VARIABLE_FLOW')
                            break

        # (16) FLOW_SCOPING
        flow_ctors = {'IF': 0, 'WHILE_LOOP': 0, 'FOR_LOOP': 1, 'ENHANCED_FOR_LOOP': 1,
                      'DO_WHILE_LOOP': 0, 'SWITCH': 0, 'TRY': 0}
        for nid, n in self.tree.nodes.items():
            if n.raw_kind not in flow_ctors:
                continue
            head_kids = self.tree.nodes[nid].child_ids[:flow_ctors[n.raw_kind] + 1]
            head_ids = set()
            stack = list(head_kids)
            while stack:
                cur = stack.pop()
                if cur in head_ids:
                    continue
                head_ids.add(cur)
                stack.extend(self.tree.nodes[cur].child_ids)
            for vid, vn in self.tree.nodes.items():
                if vn.raw_kind == 'VARIABLE' and vid in head_ids:
                    name, _ = parse_var_decl(self.node_text(vid))
                    if not name:
                        continue
                    for uid, un in self.tree.nodes.items():
                        if un.raw_kind == 'IDENTIFIER' and self.node_text(uid).strip() == name \
                                and self.is_inside(uid, nid) and uid not in head_ids:
                            add(vid, self.anchor_statement(uid), 'FLOW_SCOPING')
                            break

        # (5) VARIABLE_SHADOWING
        for name, lst in by_name.items():
            vars_ = [(sid, did) for (sid, did, kind, typ) in lst if kind == 'var']
            for i, (s1, d1) in enumerate(vars_):
                for (s2, d2) in vars_[i + 1:]:
                    if s1 != s2 and (self.is_inside(d2, s1) or self.is_inside(d1, s2)):
                        add(d1, d2, 'VARIABLE_SHADOWING')

        return edges

    # -- closure region (the extractor's closure operator, host side) -----------

    def region(self, anchor_id: int, max_hops: int = 2, max_nodes: int = MAX_REGION_NODES):
        """Region-aware closure around one anchor: roles + kept edges.

        Mirrors shannon_pipeline.closure() exactly: anchor = CORE_TRIGGER,
        statement anchors pulled in by crossing semantic edges = CONTEXT,
        external endpoints + 1-2 hop neighbors = DEPENDENT, plus enclosing
        declaration and nearest control wrapper as CONTEXT.
        """
        all_edges = self.relations
        adj = defaultdict(set)
        for e in all_edges:
            adj[e['source']].add(e['target'])
            adj[e['target']].add(e['source'])
        anchor_descendants = self.tree.descendants(anchor_id)
        region_core = anchor_descendants | {anchor_id}

        def in_core(nid: int) -> bool:
            return nid in region_core

        included: OrderedDict[int, str] = OrderedDict()
        included[anchor_id] = 'CORE_TRIGGER'

        crossing = [e for e in all_edges if in_core(e['source']) != in_core(e['target'])]
        crossing.sort(key=lambda e: (e['type'] == 'DEF_USE', e['source'], e['target']))
        frontier = {anchor_id}
        for e in crossing:
            if len(included) >= max_nodes:
                break
            for x in (e['source'], e['target']):
                if x not in included:
                    if in_core(x):
                        included[x] = 'CONTEXT'
                    else:
                        included[x] = 'DEPENDENT'
                        frontier.add(x)

        for _hop in range(1, max_hops + 1):
            nxt = set()
            for u in frontier:
                for v in adj.get(u, ()):
                    if v not in included:
                        nxt.add(v)
            for v in sorted(nxt):
                included[v] = 'DEPENDENT'
            frontier = nxt
            if len(included) >= max_nodes:
                break

        enc = self.nearest_enclosing_decl(anchor_id)
        if enc is not None and enc not in included and enc in self.tree.nodes:
            included[enc] = 'CONTEXT'
        ctl = self.nearest_ancestor(anchor_id, CONTROL_KINDS)
        if ctl is not None and ctl not in included and len(included) < max_nodes:
            included[ctl] = 'CONTEXT'
        if len(included) > max_nodes:
            core_ctx = [k for k, r in included.items() if r != 'DEPENDENT']
            dep = [k for k, r in included.items() if r == 'DEPENDENT']
            keep = core_ctx + dep[:max(0, max_nodes - len(core_ctx))]
            included = OrderedDict((k, included[k]) for k in keep)
        kept = set(included)
        edges = [e for e in all_edges if e['source'] in kept and e['target'] in kept]
        return included, edges

    # -- features ---------------------------------------------------------------

    def region_features(self, region_edges: Iterable[dict[str, Any]], node_ids: set[int]) -> set[str]:
        feats = {RELATION_TO_FEATURE[edge['type']] for edge in region_edges if edge['type'] in RELATION_TO_FEATURE}
        feats |= self.syntax_features(node_ids)
        return feats

    def syntax_features(self, node_ids: Iterable[int]) -> set[str]:
        """The four syntax-level extras from the extractor, restricted to a node set."""
        ids = set(node_ids)
        feats = set()
        # MixedType_String_Concat: a PLUS whose subtree mixes string and numeric literals
        for nid in ids:
            if self.tree.nodes[nid].raw_kind != 'PLUS':
                continue
            stack = list(self.tree.descendants(nid))
            has_str = has_num = False
            while stack and not (has_str and has_num):
                c = stack.pop()
                k = self.tree.nodes[c].raw_kind
                if k == 'STRING_LITERAL':
                    has_str = True
                elif k in ('INT_LITERAL', 'DOUBLE_LITERAL', 'LONG_LITERAL', 'FLOAT_LITERAL'):
                    has_num = True
                stack.extend(self.tree.nodes[c].child_ids)
            if has_str and has_num:
                feats.add('MixedType_String_Concat')
                break
        # Recursive_Invocation
        method_names = {}
        for nid in ids:
            if self.tree.nodes[nid].raw_kind == 'METHOD':
                nm = parse_method_name(self.node_text(nid))
                if nm:
                    method_names[nm] = nid
        if method_names:
            for nid in ids:
                node = self.tree.nodes[nid]
                if node.raw_kind == 'IDENTIFIER' and self.node_text(nid).strip() in method_names:
                    m = method_names[self.node_text(nid).strip()]
                    if self.is_inside(nid, m):
                        feats.add('Recursive_Invocation')
                        break
        if self.tree.parse_status == 'recovered_with_errors':
            feats.add('Erroneous_Parse_Recovery')
        if any(self.tree.nodes[nid].raw_kind == 'ERRONEOUS' for nid in ids):
            feats.add('Erroneous_Node_Present')
        return feats


class HostAnalysis:
    """Per-input host analysis computed once and shared by every pattern match.

    Carries the relation graph plus the Shannon subtree profile (size, kind
    counts, structural entropy) so anchor refinement uses the same entropy the
    extractor used for sink selection.
    """

    def __init__(self, tree: JavaSyntaxTree) -> None:
        self.tree = tree
        self.graph = HostGraph(tree)
        self.relations = self.graph.relations
        self.edge_index = self.graph.edge_index()
        self.relation_type_multiset: Counter[str] = Counter(edge['type'] for edge in self.relations)
        self._subtree_sizes: dict[int, int] | None = None
        self._subtree_kind_counts: dict[int, Counter[str]] | None = None
        self._subtree_entropy: dict[int, float] | None = None

    # -- Shannon subtree profile (mirrors the extractor's aggregates) -----------

    def _ensure_profile(self) -> None:
        if self._subtree_sizes is not None:
            return
        tree = self.tree
        sizes: dict[int, int] = {}
        counts: dict[int, Counter[str]] = {}
        depths: dict[int, int] = {}
        entropies: dict[int, float] = {}
        stack = [(tree.root_id, False)]
        while stack:
            nid, done = stack.pop()
            node = tree.nodes[nid]
            if not done:
                stack.append((nid, True))
                for child in node.child_ids:
                    stack.append((child, False))
                continue
            local = Counter({node.raw_kind: 1})
            size, depth = 1, 0
            for child in node.child_ids:
                local.update(counts.get(child) or {})
                size += sizes.get(child, 1)
                depth = max(depth, depths.get(child, 0) + 1)
            sizes[nid] = size
            counts[nid] = local
            depths[nid] = depth
            if size >= 2:
                h = -sum((v / size) * math.log2(v / size) for v in local.values())
                h_norm = h / math.log2(size)
            else:
                h_norm = 0.0
            richness = len(local)
            se = h_norm + 0.5 * min(depth, 10) / 10 + 0.5 * min(richness, 20) / 20
            if node.raw_kind == 'ERRONEOUS':
                se *= 1.25
            entropies[nid] = se
        self._subtree_sizes = sizes
        self._subtree_kind_counts = counts
        self._subtree_entropy = entropies

    def subtree_size(self, node_id: int) -> int:
        self._ensure_profile()
        return self._subtree_sizes.get(node_id, 1)  # type: ignore[union-attr]

    def subtree_kind_counts(self, node_id: int) -> Counter[str]:
        self._ensure_profile()
        return self._subtree_kind_counts.get(node_id, Counter())  # type: ignore[union-attr]

    def subtree_entropy(self, node_id: int) -> float:
        self._ensure_profile()
        return self._subtree_entropy.get(node_id, 0.0)  # type: ignore[union-attr]

    # -- region helpers -----------------------------------------------------------

    def anchor_span(self, anchor_id: int) -> tuple[int, int]:
        node = self.tree.nodes[anchor_id]
        return node.start_byte, node.end_byte

    def region_signature(self, roles: dict[int, str], edges: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, int], set[str]]:
        """(kind_role_multiset, edge_type_multiset, features) for one closure region."""
        weights = {'CORE_TRIGGER': 1.0, 'CONTEXT': 0.6, 'DEPENDENT': 0.3}
        kinds: Counter[str] = Counter()
        for nid, role in roles.items():
            kinds[self.tree.nodes[nid].raw_kind] += weights.get(role, 0.3)
        edge_types: Counter[str] = Counter(edge['type'] for edge in edges)
        feats = self.graph.region_features(edges, set(roles))
        return dict(sorted(kinds.items())), dict(sorted(edge_types.items())), feats
