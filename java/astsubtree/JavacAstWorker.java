package astsubtree;

import com.sun.source.tree.CompilationUnitTree;
import com.sun.source.tree.LineMap;
import com.sun.source.tree.Tree;
import com.sun.source.util.JavacTask;
import com.sun.source.util.SourcePositions;
import com.sun.source.util.TreeScanner;
import com.sun.source.util.Trees;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import javax.tools.Diagnostic;
import javax.tools.DiagnosticCollector;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;

/** Emits a compact TSV AST using only the public javac Tree API. */
public final class JavacAstWorker {
    private JavacAstWorker() {}

    private static final class ParentFrame {
        final int nodeId;
        int nextChildIndex;

        ParentFrame(int nodeId) {
            this.nodeId = nodeId;
        }
    }

    private static final class Emitter extends TreeScanner<Void, ParentFrame> {
        private final SourcePositions positions;
        private CompilationUnitTree unit;
        private LineMap lineMap;
        private int nextNodeId;

        Emitter(SourcePositions positions) {
            this.positions = positions;
        }

        void emit(CompilationUnitTree compilationUnit) {
            unit = compilationUnit;
            lineMap = unit.getLineMap();
            scan(unit, null);
        }

        @Override
        public Void scan(Tree tree, ParentFrame parent) {
            if (tree == null) {
                return null;
            }
            int nodeId = nextNodeId++;
            int parentId = parent == null ? -1 : parent.nodeId;
            int childIndex = parent == null ? 0 : parent.nextChildIndex++;
            long start = positions.getStartPosition(unit, tree);
            long end = positions.getEndPosition(unit, tree);
            long startLine = line(start);
            long endLine = line(end > start ? end - 1 : start);
            System.out.printf(
                    "N\t%d\t%s\t%d\t%d\t%d\t%d\t%d\t%d%n",
                    nodeId,
                    tree.getKind().name(),
                    parentId,
                    childIndex,
                    start,
                    end,
                    startLine,
                    endLine);
            return super.scan(tree, new ParentFrame(nodeId));
        }

        private long line(long position) {
            return position < 0 || lineMap == null ? -1 : lineMap.getLineNumber(position);
        }
    }

    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            System.err.println("usage: JavacAstWorker SOURCE.java");
            System.exit(2);
        }
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        if (compiler == null) {
            throw new IllegalStateException("system Java compiler is unavailable");
        }
        DiagnosticCollector<JavaFileObject> diagnostics = new DiagnosticCollector<>();
        try (StandardJavaFileManager manager =
                compiler.getStandardFileManager(diagnostics, null, StandardCharsets.UTF_8)) {
            Iterable<? extends JavaFileObject> files =
                    manager.getJavaFileObjectsFromPaths(List.of(Path.of(args[0])));
            JavacTask task = (JavacTask)
                    compiler.getTask(null, manager, diagnostics, List.of("-proc:none"), null, files);
            Iterable<? extends CompilationUnitTree> units = task.parse();
            long errorCount = diagnostics.getDiagnostics().stream()
                    .filter(item -> item.getKind() == Diagnostic.Kind.ERROR)
                    .count();
            System.out.printf("S\t%s\t%d%n", errorCount == 0 ? "parsed" : "recovered_with_errors", errorCount);
            Emitter emitter = new Emitter(Trees.instance(task).getSourcePositions());
            for (CompilationUnitTree unit : units) {
                emitter.emit(unit);
            }
        }
    }
}

