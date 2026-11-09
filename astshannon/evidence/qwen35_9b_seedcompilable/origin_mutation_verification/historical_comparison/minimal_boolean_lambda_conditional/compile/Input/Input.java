class Test { Object foo() { return true ? bar : (boolean) (() -> false); } }
