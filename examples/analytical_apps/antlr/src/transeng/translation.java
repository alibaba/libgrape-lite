
//Translate.java

import org.antlr.v4.gui.TreeViewer;
import org.antlr.v4.runtime.ANTLRInputStream;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.tree.ParseTree;

import javax.swing.*;
import java.io.*;
import java.util.Arrays;
import java.util.Scanner;
@SuppressWarnings("unchecked")
public class translation {
	static int stage=0;//处理阶段，当stage=0时，识别agg和gen，当stage=1时，解析agg和gen
	static String outputfile="../choose_egine/testz3.java";
	static int algorithom_select=0;//0--select_sssp || 1--select_cc || 2--select_pagerank ||3--select_php

	public static ParseTree run(String expr) throws Exception {
		// 对每一个输入的字符串，构造一个 ANTLRStringStream 流 in
		ANTLRInputStream input = new ANTLRInputStream(expr);
		// 用 in 构造词法分析器 lexer，词法分析的作用是将字符聚集成单词或者符号
		CPP14Lexer lexer = new CPP14Lexer(input);
		// 用词法分析器 lexer 构造一个记号流 tokens
		CommonTokenStream tokens = new CommonTokenStream(lexer);
		// 再使用 tokens 构造语法分析器 parser,至此已经完成词法分析和语法分析的准备工作
		CPP14Parser parser = new CPP14Parser(tokens);
		// 最终调用语法分析器的规则 translationUnit（这个是我们在CPP14.g4里面定义的那个规则）
		ParseTree tree = parser.translationUnit();

		if(stage==0) {
			funcVisitor visitor = new funcVisitor();
			visitor.visit(tree);
			// System.out.println();
			return tree;
		}
		else if(stage==1)
		{
			addExpVisitor visitor = new addExpVisitor();
			visitor.visit(tree);
			// System.out.println();
		}
		// JFrame frame = new JFrame("ANTLR AST");
		// JPanel panel = new JPanel();
		// TreeViewer viewer = new TreeViewer(Arrays.asList(parser.getRuleNames()), tree);
		//  viewer.setScale(0.7); // Scale a little
		//  panel.add(viewer);
		//  frame.add(panel);
		//  frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		//  frame.pack();
		//  frame.setVisible(true);
		return tree;
	}


	public static void main(String[] args) throws Exception {
		stage=0;
		String user_app = args[0];
		String str0 =fileRead(user_app);
		alg_select(user_app);
		StringBuffer sb0 = new StringBuffer();
		funcVisitor.agg_Str=sb0.append(str0).substring(str0.indexOf("bool accumulate"),str0.indexOf("}",str0.indexOf("bool accumulate"))+1);
		funcVisitor.gen_Str=sb0.append(str0).substring(str0.indexOf("value_t generate"),str0.indexOf("}",str0.indexOf("value_t generate"))+1);
		// System.out.println("agg_stri is : "+funcVisitor.agg_Str);
		// System.out.println("gen_str is : "+funcVisitor.gen_Str);

		// System.out.println("**********************************************************************************************");
		stage=1;
		String str1 =fileRead(outputfile);
		StringBuffer sb = new StringBuffer();
		switch (alg_select(user_app)) {
			case 0:
			   sb.append(str1).replace(str1.indexOf("public Expr agg("),+str1.indexOf("}",str1.indexOf("public Expr gen("))+1,ssspAggFactory(funcVisitor.agg_Str)+ssspGenFactory(funcVisitor.gen_Str));
			   break;
			case 1:
			   sb.append(str1).replace(str1.indexOf("public Expr agg("),+str1.indexOf("}",str1.indexOf("public Expr gen("))+1,ccAggFactory(funcVisitor.agg_Str)+ccGenFactory(funcVisitor.gen_Str));
			   break;
			case 2:
			   sb.append(str1).replace(str1.indexOf("public Expr agg("),+str1.indexOf("}",str1.indexOf("public Expr gen("))+1,pgAggFactory(funcVisitor.agg_Str)+pgGenFactory(funcVisitor.gen_Str));
			   break;
			case 3:
			   sb.append(str1).replace(str1.indexOf("public Expr agg("),+str1.indexOf("}",str1.indexOf("public Expr gen("))+1,phpAggFactory(funcVisitor.agg_Str)+phpGenFactory(funcVisitor.gen_Str));
			   break;
			case 4:
					sb.append(str1).replace(str1.indexOf("public Expr agg("),+str1.indexOf("}",str1.indexOf("public Expr gen("))+1,sswpAggFactory(funcVisitor.agg_Str)+sswpGenFactory(funcVisitor.gen_Str));
			   break;
		 }
		//System.out.println(sb.toString());
		try {
			FileWriter fw=new FileWriter("../choose_egine/testz3.java");
			fw.write(sb.toString());
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
			return;

	}

	public static String pgAggFactory(String agg_str) throws Exception {
		addExpVisitor.analysis_mode=0;
		//System.out.println("Input: " + agg_str);
		run(agg_str);
		expr.myMulExpr myMExpr=new expr.myMulExpr();
		myMExpr.MulExprFactory(expr.MULoperators, expr.MULexpr);
		expr.myAddExpr myAExpr=new expr.myAddExpr();
		myAExpr.AddExprFactory(expr.ADDoperators, expr.ADDexpr);
		expr.myAtomAddExpr myAtomAExpr=new expr.myAtomAddExpr();
		myAtomAExpr.AtomAddExprFactory(expr.AtomADDoperators, expr.AtomADDexpr);
		String output="\npublic Expr agg(Expr "+ expr.AtomExprList.get(0).myAtomADDexpr.get(0)+",Expr "+
				expr.AtomExprList.get(0).myAtomADDexpr.get(1)+",Context check_gfg) { \n\t\t\tExpr c = check_gfg.mkAdd("
				+ expr.AtomExprList.get(0).myAtomADDexpr.get(0)+","+ expr.AtomExprList.get(0).myAtomADDexpr.get(1)
				+");\n\t\t\t return c;\n} ";
		return output;
	}
	public static String pgGenFactory(String gen_str) throws Exception {
		addExpVisitor.analysis_mode=1;
			// System.out.println("Input: " + gen_str);
			run(gen_str);
		expr.myMulExpr myMExpr=new expr.myMulExpr();
		myMExpr.MulExprFactory(expr.MULoperators, expr.MULexpr);
		expr.myAddExpr myAExpr=new expr.myAddExpr();
		myAExpr.AddExprFactory(expr.ADDoperators, expr.ADDexpr);
		expr.myAtomAddExpr myAtomAExpr=new expr.myAtomAddExpr();
		myAtomAExpr.AtomAddExprFactory(expr.AtomADDoperators, expr.AtomADDexpr);

		String output="\npublic Expr gen(Expr x2,Solver s,Context check_gfg) {\n"
			+"\tExpr d = check_gfg.mkRealConst(\""+expr.AddExprList.get(0).myADDexpr.get(0).myMulExpr.get(1)+"\");\n"
			+"\tExpr w = check_gfg.mkRealConst(\""+expr.AddExprList.get(0).myADDexpr.get(0).myMulExpr.get(2)+"\");\n"
			+"\tExpr val = check_gfg.mkMul(x2,d,w);\n"
			+"\treturn val;\n}	";
		// System.out.println(output);
		return output;
	}

	public static String ccAggFactory(String agg_str) throws Exception {
		addExpVisitor.analysis_mode=1;
		// System.out.println("Input: " + agg_str);
		run(agg_str);

		expr.myMulExpr myMExpr=new expr.myMulExpr();
		myMExpr.MulExprFactory(expr.MULoperators, expr.MULexpr);
		expr.myAddExpr myAExpr=new expr.myAddExpr();
		myAExpr.AddExprFactory(expr.ADDoperators, expr.ADDexpr);
		expr.myAtomAddExpr myAtomAExpr=new expr.myAtomAddExpr();
		myAtomAExpr.AtomAddExprFactory(expr.AtomADDoperators, expr.AtomADDexpr);


		String output="\npublic Expr agg(Expr "+ expr.AtomExprList.get(0).myAtomADDexpr.get(0)+",Expr "+
				expr.AtomExprList.get(0).myAtomADDexpr.get(1)+",Context check_gfg) { \n\t\t\tBoolExpr f = check_gfg.mkLt("+expr.AtomExprList.get(0).myAtomADDexpr.get(0)+","+ expr.AtomExprList.get(0).myAtomADDexpr.get(1)
				+");\n\t\t\tExpr ite = check_gfg.mkITE(f,"
				+ expr.AtomExprList.get(0).myAtomADDexpr.get(0)+","+ expr.AtomExprList.get(0).myAtomADDexpr.get(1)
				+");\n\t\t\t return ite;\n} ";
		// System.out.println(output);
		return output;
	}
	public static String ccGenFactory(String gen_str) throws Exception {
		addExpVisitor.analysis_mode=1;
		// System.out.println("Input: " + gen_str);
		run(gen_str);

		expr.myMulExpr myMExpr=new expr.myMulExpr();
		myMExpr.MulExprFactory(expr.MULoperators, expr.MULexpr);
		expr.myAddExpr myAExpr=new expr.myAddExpr();
		myAExpr.AddExprFactory(expr.ADDoperators, expr.ADDexpr);
		expr.myAtomAddExpr myAtomAExpr=new expr.myAtomAddExpr();
		myAtomAExpr.AtomAddExprFactory(expr.AtomADDoperators, expr.AtomADDexpr);
	
		String output="\npublic Expr gen(Expr "+expr.AddExprList.get(0).myADDexpr.get(0).myMulExpr.get(0)+",Solver s,Context check_gfg) {\n"
				+"\tExpr val = "+expr.AddExprList.get(0).myADDexpr.get(0).myMulExpr.get(0)+";\n"
				+"\treturn val;\n}	";
		return output;
	}

	public static String ssspAggFactory(String agg_str) throws Exception {
		addExpVisitor.analysis_mode=1;
		// System.out.println("Input: " + agg_str);
		run(agg_str);
		expr.myMulExpr myMExpr=new expr.myMulExpr();
		myMExpr.MulExprFactory(expr.MULoperators, expr.MULexpr);
		expr.myAddExpr myAExpr=new expr.myAddExpr();
		myAExpr.AddExprFactory(expr.ADDoperators, expr.ADDexpr);
		expr.myAtomAddExpr myAtomAExpr=new expr.myAtomAddExpr();
		myAtomAExpr.AtomAddExprFactory(expr.AtomADDoperators, expr.AtomADDexpr);
		String output="\npublic Expr agg(Expr "+ expr.AtomExprList.get(0).myAtomADDexpr.get(0)+",Expr "+
				expr.AtomExprList.get(0).myAtomADDexpr.get(1)+",Context check_gfg) { \n\t\t\tBoolExpr f = check_gfg.mkLt("+expr.AtomExprList.get(0).myAtomADDexpr.get(0)+","+ expr.AtomExprList.get(0).myAtomADDexpr.get(1)+");\n\t\t\tExpr ite = check_gfg.mkITE(f,"
				+ expr.AtomExprList.get(0).myAtomADDexpr.get(0)+","+ expr.AtomExprList.get(0).myAtomADDexpr.get(1)
				+");\n\t\t\t return ite;\n} ";
		// System.out.println(output);
		return output;
	}
	public static String ssspGenFactory(String gen_str) throws Exception {
		addExpVisitor.analysis_mode=1;
		// System.out.println("Input: " + gen_str);
		run(gen_str);
		expr.myMulExpr myMExpr=new expr.myMulExpr();
		myMExpr.MulExprFactory(expr.MULoperators, expr.MULexpr);
		expr.myAddExpr myAExpr=new expr.myAddExpr();
		myAExpr.AddExprFactory(expr.ADDoperators, expr.ADDexpr);
		expr.myAtomAddExpr myAtomAExpr=new expr.myAtomAddExpr();
		myAtomAExpr.AtomAddExprFactory(expr.AtomADDoperators, expr.AtomADDexpr);

		String output="\npublic Expr gen(Expr "+expr.AddExprList.get(0).myADDexpr.get(0).myMulExpr.get(0)+",Solver s,Context check_gfg) {\n"
				+"\tExpr dis = check_gfg.mkRealConst(\"dis\");\n"
				+"\tExpr val = check_gfg.mkAdd("+expr.AddExprList.get(0).myADDexpr.get(0).myMulExpr.get(0)+",dis);\n"
				+"\treturn val;\n}	";
		return output;
	}
	public static String sswpAggFactory(String agg_str) throws Exception {
		addExpVisitor.analysis_mode=1;
		// System.out.println("Input: " + agg_str);
		run(agg_str);
		expr.myMulExpr myMExpr=new expr.myMulExpr();
		myMExpr.MulExprFactory(expr.MULoperators, expr.MULexpr);
		expr.myAddExpr myAExpr=new expr.myAddExpr();
		myAExpr.AddExprFactory(expr.ADDoperators, expr.ADDexpr);
		expr.myAtomAddExpr myAtomAExpr=new expr.myAtomAddExpr();
		myAtomAExpr.AtomAddExprFactory(expr.AtomADDoperators, expr.AtomADDexpr);
		String output="\npublic Expr agg(Expr "+ expr.AtomExprList.get(0).myAtomADDexpr.get(0)+",Expr "+
				expr.AtomExprList.get(0).myAtomADDexpr.get(1)+",Context check_gfg) { \n\t\t\tBoolExpr f = check_gfg.mkLt("+expr.AtomExprList.get(0).myAtomADDexpr.get(0)+","+ expr.AtomExprList.get(0).myAtomADDexpr.get(1)+");\n\t\t\tExpr ite = check_gfg.mkITE(f,"
				+ expr.AtomExprList.get(0).myAtomADDexpr.get(0)+","+ expr.AtomExprList.get(0).myAtomADDexpr.get(1)
				+");\n\t\t\t return ite;\n} ";
		// System.out.println(output);
		return output;
	}
	public static String sswpGenFactory(String gen_str) throws Exception {
		addExpVisitor.analysis_mode=1;
		run(gen_str);
		// System.out.println("sswp gen :"+gen_str);
		expr.myMulExpr myMExpr=new expr.myMulExpr();
		myMExpr.MulExprFactory(expr.MULoperators, expr.MULexpr);
		expr.myAddExpr myAExpr=new expr.myAddExpr();
		myAExpr.AddExprFactory(expr.ADDoperators, expr.ADDexpr);
		expr.myAtomAddExpr myAtomAExpr=new expr.myAtomAddExpr();
		myAtomAExpr.AtomAddExprFactory(expr.AtomADDoperators, expr.AtomADDexpr);
		// System.out.println("first : "+expr.AddExprList.get(2).myADDexpr.get(0).myMulExpr.get(0)+expr.AddExprList.get(1).myADDexpr.get(0).myMulExpr.get(0));
		String output="\npublic Expr gen(Expr "+expr.AddExprList.get(2).myADDexpr.get(0).myMulExpr.get(0)+",Solver s,Context check_gfg) {\n"
				+"\tExpr v = check_gfg.mkRealConst(\"v\");\n"
				+"\tBoolExpr f = check_gfg.mkLt("+expr.AddExprList.get(2).myADDexpr.get(0).myMulExpr.get(0)+","+expr.AddExprList.get(1).myADDexpr.get(0).myMulExpr.get(0)+");\n"+"\tExpr val = check_gfg.mkITE( f,"+expr.AddExprList.get(2).myADDexpr.get(0).myMulExpr.get(0)+","+expr.AddExprList.get(1).myADDexpr.get(0).myMulExpr.get(0)+" );\n"
				+"\treturn val;\n}	";
				// System.out.println(output);
		return output;
	}
	public static String phpAggFactory(String agg_str) throws Exception {
		addExpVisitor.analysis_mode=0;
		// System.out.println("Input: " + agg_str);
		run(agg_str);
		expr.myMulExpr myMExpr=new expr.myMulExpr();
		myMExpr.MulExprFactory(expr.MULoperators, expr.MULexpr);
		expr.myAddExpr myAExpr=new expr.myAddExpr();
		myAExpr.AddExprFactory(expr.ADDoperators, expr.ADDexpr);
		expr.myAtomAddExpr myAtomAExpr=new expr.myAtomAddExpr();
		myAtomAExpr.AtomAddExprFactory(expr.AtomADDoperators, expr.AtomADDexpr);


		String output="\npublic Expr agg(Expr "+ expr.AtomExprList.get(0).myAtomADDexpr.get(0)+",Expr "+
				expr.AtomExprList.get(0).myAtomADDexpr.get(1)+",Context check_gfg) { \n\t\t\tExpr c = check_gfg.mkAdd("
				+ expr.AtomExprList.get(0).myAtomADDexpr.get(0)+","+ expr.AtomExprList.get(0).myAtomADDexpr.get(1)
				+");\n\t\t\t return c;\n} ";
		return output;
	}
	public static String phpGenFactory(String gen_str) throws Exception {
		addExpVisitor.analysis_mode=1;
		// System.out.println("Input: " + gen_str);
		run(gen_str);
		expr.myMulExpr myMExpr=new expr.myMulExpr();
		myMExpr.MulExprFactory(expr.MULoperators, expr.MULexpr);
		expr.myAddExpr myAExpr=new expr.myAddExpr();
		myAExpr.AddExprFactory(expr.ADDoperators, expr.ADDexpr);
		expr.myAtomAddExpr myAtomAExpr=new expr.myAtomAddExpr();
		myAtomAExpr.AtomAddExprFactory(expr.AtomADDoperators, expr.AtomADDexpr);

		String output="\npublic Expr gen(Expr x2,Solver s,Context check_gfg) {\n"
				+"\tExpr d = check_gfg.mkRealConst(\""+expr.AddExprList.get(0).myADDexpr.get(0).myMulExpr.get(1)+"\");\n"
				+"\tExpr w = check_gfg.mkRealConst(\""+expr.AddExprList.get(0).myADDexpr.get(0).myMulExpr.get(2)+"\");\n"
				+"\tExpr val = check_gfg.mkMul(x2,d,w);\n"
				+"\treturn val;\n}	";
		// System.out.println(output);
		return output;
	}

	public static String fileRead(String fileName) throws Exception {
		File file = new File(fileName);//定义一个file对象，用来初始化FileReader
		FileReader reader = new FileReader(file);//定义一个fileReader对象，用来初始化BufferedReader
		BufferedReader bReader = new BufferedReader(reader);//new一个BufferedReader对象，将文件内容读取到缓存
		StringBuilder sb = new StringBuilder();//定义一个字符串缓存，将字符串存放缓存中
		String s = "";
		while ((s =bReader.readLine()) != null) {//逐行读取文件内容，不读取换行符和末尾的空格
			sb.append(s + "\n");//将读取的字符串添加换行符后累加存放在缓存中
	//		System.out.println(s);
		}
		bReader.close();
		String str = sb.toString();
		return str;
	}
	public static int alg_select(String fileName) throws Exception {//算法识别
		if(fileName.equals("../../../sssp/sssp_ingress.h")&fileName.equals("../../../bfs/bfs_ingress.h"))
			algorithom_select=0;
		else if(fileName.equals("../../../cc/cc_ingress.h"))
			algorithom_select=1;
		else if(fileName.equals("../../../pagerank/pagerank_ingress.h"))
			algorithom_select=2;
		else if(fileName.equals("../../../php/php_ingress.h"))
			algorithom_select=3;
		else if(fileName.equals("../../../sswp/sswp_ingress.h"))
			algorithom_select=4;
			return algorithom_select;
	}
}