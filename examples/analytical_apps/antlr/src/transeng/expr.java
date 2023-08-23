import java.util.Vector;
@SuppressWarnings("unchecked")
public class expr {

    public static int MULexprinADDexprNum=0;
    public static int PriexprinADDexprNum=0;
    public static int AtomParaNum=0;

    private static int cout=50;//以下各个二维数组的列代表第几个表达式，行中包含了表达式中的元素

    public  static  String[][] ADDoperators = new String[cout][cout];//加法表达式的操作符
    public  static  String[][] ADDexpr = new String[cout][cout];//加法表达式的各个操作数即是MULexpr
    public  static  String[][] MULoperators = new String[cout][cout];//乘法表达式的操作符
    public  static  String[][] MULexpr = new String[cout][cout];//乘法表达式的各个部分
    public  static  String[][] AtomADDoperators = new String[cout][cout];//原子操作的操作符
    public  static  String[][] AtomADDexpr = new String[cout][cout];//原子操作的各个操作数即是MULexpr

    public  static Vector<myAddExpr> AddExprList=new Vector<>();//add的解析
    public  static Vector<myMulExpr> MulExprList=new Vector<>();//mul的解析
    public  static Vector<myAtomAddExpr> AtomExprList=new Vector<>();//原子操作的解析



public static void clearLists(){
    AddExprList.clear();
    MulExprList.clear();
    AtomExprList.clear();
    ADDoperators = new String[cout][cout];//加法表达式的操作符
    ADDexpr = new String[cout][cout];//加法表达式的各个操作数即是MULexpr
    MULoperators = new String[cout][cout];//乘法表达式的操作符
    MULexpr = new String[cout][cout];//乘法表达式的各个部分
    AtomADDoperators = new String[cout][cout];//原子操作的操作符
    AtomADDexpr = new String[cout][cout];//原子操作的各个操作数即是MULexpr
}





    public static class myAddExpr{
        Vector<String> myADDoperators=new Vector<>();
        Vector<myMulExpr> myADDexpr=new Vector<>();
        public void AddExprFactory(String[][] ADDoperators,String[][] ADDexpr)
        {
            int mulnum=0;
            for(int i = 0; i< addExpVisitor.addtimes; i++)
            {
                int k=0;
                int j=1;
                myAddExpr myaddexpr=new myAddExpr();
                while(ADDexpr[i][k]!=null)
                {
                    myaddexpr.myADDexpr.add(MulExprList.get(mulnum));
                    mulnum++;
                    k++;
                }
                while(ADDoperators[i][j]!=null)
                {
                    myaddexpr.myADDoperators.add(ADDoperators[i][j]);
                    j=j+2;
                }
                AddExprList.add(myaddexpr);
            }
        }
    }
    public static class myAtomAddExpr{
        Vector<String> myAtomADDoperators=new Vector<>();
        Vector<String> myAtomADDexpr=new Vector<>();
        public void AtomAddExprFactory(String[][] AtomADDoperators,String[][] AtomADDexpr)
        {
            for(int i = 0; i< addExpVisitor.atomtimes; i++)
            {
                int k=0;
                int j=1;
                myAtomAddExpr myatomaddExpr=new myAtomAddExpr();
                while(AtomADDexpr[i][k]!=null)
                {
                    myatomaddExpr.myAtomADDexpr.add(AtomADDexpr[i][k]);
                    k++;
                }
                while(AtomADDoperators[i][j]!=null)
                {
                    myatomaddExpr.myAtomADDoperators.add(AtomADDoperators[i][j]);
                    j=j+2;
                }
                AtomExprList.add(myatomaddExpr);
            }
        }
    }
    public static class myMulExpr{
        Vector<String> myMULoperators=new Vector<>();
        Vector<String> myMulExpr=new Vector<>();
        public void MulExprFactory(String[][] MULoperators,String[][] MULexpr)
        {

            for(int i = 0; i< addExpVisitor.multimes; i++)
            {
                int k=0;
                int j=1;
                myMulExpr mymulexpr=new myMulExpr();
                while(MULexpr[i][k]!=null)
                {
                    mymulexpr.myMulExpr.add(MULexpr[i][k]);
                    k++;
                }
                while(MULoperators[i][j]!=null)
                {
                    mymulexpr.myMULoperators.add(MULoperators[i][j]);
                    j=j+2;
                }
                MulExprList.add(mymulexpr);
            }

        }

    }
}
