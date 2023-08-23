import java.util.List;
import java.util.ListIterator;
import java.util.Map;

import org.antlr.v4.misc.OrderedHashMap;
import org.antlr.v4.runtime.Parser;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.ParseTreeListener;
import org.antlr.v4.runtime.tree.TerminalNode;
@SuppressWarnings("unchecked")
public class addExpVisitor extends CPP14ParserBaseVisitor<Object> {


    public  static int addtimes=0;//加法运算的次数
    public static int multimes = 0;//乘法运算的次数
    public static int atomtimes = 0;//原子运算的次数
    public static int atom_analysis=0;
    public static int atom_analysis_paraNum=0;
    public static int analysis_mode=1;//解析模式：1为表达式解析 0为return不需要解析


    @Override
    public Object visitAdditiveExpression(CPP14Parser.AdditiveExpressionContext ctx)
    {
        if(analysis_mode==0||atom_analysis>0)
            return visitChildren(ctx);
        else{
        expr.MULexprinADDexprNum=0;
        int childNum=ctx.getChildCount();
        for(int i=1;i<childNum;i=i+2) {
            expr.ADDoperators[addtimes][i]= String.valueOf(ctx.getChild(i));
        }
        addtimes++;
        return visitChildren(ctx);}
    }

    @Override
    public Object visitMultiplicativeExpression(CPP14Parser.MultiplicativeExpressionContext ctx) {
        if(analysis_mode==0||atom_analysis>0)
            return visitChildren(ctx);
        else{
        expr.PriexprinADDexprNum=0;
        expr.ADDexpr[addtimes-1][expr.MULexprinADDexprNum]=ctx.getText();
        expr.MULexprinADDexprNum++;
        int childNum= ctx.getChildCount();
        for(int i=1;i<childNum;i=i+2) {
            expr.MULoperators[multimes][i]= String.valueOf(ctx.getChild(i));
        }
        multimes++;
//        System.out.println("MULchildurnNum:" + ctx.getChildCount());
        List<ParseTree> NoTnodes;
        NoTnodes=ctx.children;
//        System.out.println("NoTnodes-Mulexp:" + NoTnodes);
        return visitChildren(ctx);}
    }

    @Override
    public Object visitPrimaryExpression(CPP14Parser.PrimaryExpressionContext ctx) {
        if(analysis_mode==0||atom_analysis>0)
            return visitChildren(ctx);
        else{
        String Tnodes;
        Tnodes=ctx.getText();
        if(Tnodes.equals("atomic_add")||Tnodes.equals("atomic_min")) {
            addtimes--;
            multimes--;
            atom_analysis = 1;
            return visitChildren(ctx);
        }
        expr.MULexpr[multimes-1][expr.PriexprinADDexprNum]=Tnodes;
        expr.PriexprinADDexprNum++;
        return visitChildren(ctx);}
    }


    @Override
    public Object visitLiteral(CPP14Parser.LiteralContext ctx) {
        if(analysis_mode==0||atom_analysis>0)
            return visitChildren(ctx);
        else{
//        System.out.print("Literal:");
        String NTnodes;
        NTnodes=ctx.toString();

        return visitChildren(ctx);}
    }

    @Override public Object visitNoPointerDeclarator(CPP14Parser.NoPointerDeclaratorContext ctx)
    {     return visitChildren(ctx); }

    @Override public Object visitIdExpression(CPP14Parser.IdExpressionContext ctx)
    {
        String NTnodes;
        NTnodes=ctx.getText();
        if(NTnodes.equals("atomic_add")||NTnodes.equals("atomic_min"))
            atom_analysis=1;
        return visitChildren(ctx);
    }

    @Override public Object visitParameterDeclaration(CPP14Parser.ParameterDeclarationContext ctx) {
        String NTnodes;
        NTnodes=ctx.getText();
        if(atom_analysis_paraNum==2) {
            expr.AtomParaNum=0;
            atom_analysis = 0;
            return visitChildren(ctx);
        }
        if(atom_analysis==1){
            atomtimes++;
            // System.out.println(NTnodes);
            atom_analysis_paraNum++;
            expr.AtomADDoperators[atomtimes-1][1]="+";
            expr.AtomADDexpr[atomtimes-1][expr.AtomParaNum]=ctx.getText();
            expr.AtomParaNum++;
            atom_analysis++;
            return visitChildren(ctx);
        }
        if(atom_analysis==2){
            // System.out.println(NTnodes);
            atom_analysis_paraNum++;
            expr.AtomADDoperators[atomtimes-1][1]="+";
            expr.AtomADDexpr[atomtimes-1][expr.AtomParaNum]=ctx.getText();
            expr.AtomParaNum++;
            return visitChildren(ctx);
        }
        return visitChildren(ctx); }

    @Override public Object visitAssignmentExpression(CPP14Parser.AssignmentExpressionContext ctx) {
        String NTnodes;
        NTnodes=ctx.getText();
        if(atom_analysis_paraNum==2) {
            expr.AtomParaNum=0;
            atom_analysis = 0;
            return visitChildren(ctx);
        }
        if(atom_analysis==1){
            atomtimes++;
            // System.out.println(NTnodes);
            atom_analysis_paraNum++;
            expr.AtomADDoperators[atomtimes-1][1]="+";
            expr.AtomADDexpr[atomtimes-1][expr.AtomParaNum]=ctx.getText();
            expr.AtomParaNum++;
            atom_analysis++;
            return visitChildren(ctx);
        }
        if(atom_analysis==2){
            // System.out.println(NTnodes);
            atom_analysis_paraNum++;
            expr.AtomADDoperators[atomtimes-1][1]="+";
            expr.AtomADDexpr[atomtimes-1][expr.AtomParaNum]=ctx.getText();
            expr.AtomParaNum++;
            return visitChildren(ctx);
        }
        return visitChildren(ctx);
    }


}
