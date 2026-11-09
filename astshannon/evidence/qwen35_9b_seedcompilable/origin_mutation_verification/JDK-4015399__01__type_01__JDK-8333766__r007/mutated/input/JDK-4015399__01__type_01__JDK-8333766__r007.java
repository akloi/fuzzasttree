class locnames {
	    public class Test1 {
	        class Inner {
	            Inner() {
	                this(() -> new Object() { { Test1.this.hashCode(); } });
	            }
	            Inner(Runnable r) {
	            }
	        }
	    }
	    void bar() { class C {} }
	}
