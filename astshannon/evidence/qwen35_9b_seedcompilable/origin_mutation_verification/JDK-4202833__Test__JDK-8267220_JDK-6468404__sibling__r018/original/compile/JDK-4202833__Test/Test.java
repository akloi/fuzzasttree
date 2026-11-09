public class Test {

   String value;

   public Test( String pValue ){
      value = pValue;
   }

   private Test( Test pAnother ){
      value = pAnother.value;
   }

   public String getValue() {
      return value;
   }

   public Test getModified() {

      return new Test( this ){
         public String getValue() {
            return "MOD:" + super.getValue();
         }
      };
   }
}
