package testcases.logger.file;




public class Test1 {

    protected Test1() {
   	
    }

    /* (non-Javadoc)
     * @see testcases.logger.ExpLogger#addExceptionToLog(java.lang.Exception)
     */
    public void addExceptionToLog(final Exception exp) {
		
    }
    // Commented because of bug in compiler.

    public static Test1 getFileExceptionLogger() {
	return new Test1() ;
    }

    /* (non-Javadoc)
     * @see testcases.logger.ExpLogger#addExceptionTrace(java.lang.Exception)
     */

    public void addExceptionTrace(final Exception exp) {
		
    }

    /* (non-Javadoc)
     * @see testcases.logger.TraceLevel#getTraceLevel(int)
     */

    public int getTraceLevel(int level) {
	return ((true) ? 0 : ((Double) (() -> 10.0)));
    }

    /* (non-Javadoc)
     * @see testcases.logger.TraceLevel#isMax(int)
     */
    public boolean isMax(int level) {
	return false;
    }

}

