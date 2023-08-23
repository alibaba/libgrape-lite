import java.util.HashMap;
import com.microsoft.z3.*;

//import org.antlr.v4.runtime.misc.ObjectEqualityComparator;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;

public class Usage {
	public boolean check_gfg_z3(Solver s,Context check_ctx,testz3 test) {
		
		//System.out.println("check gfg execution");
		RealExpr x1=check_ctx.mkRealConst("x1");
        RealExpr x2=check_ctx.mkRealConst("x2");
	    RealExpr y1=check_ctx.mkRealConst("y1");
	    RealExpr y2=check_ctx.mkRealConst("y2");
		Expr[] bound = new Expr[]{x1,x2,y1,y2};
		Expr a=test.agg(test.gen(test.agg(x1, y1, check_ctx), s, check_ctx), test.gen(test.agg(x2, y2, check_ctx), s, check_ctx), check_ctx);
		Expr b = test.agg(test.agg(test.agg(test.gen(x1, s, check_ctx), 
				test.gen(y1, s, check_ctx), check_ctx), test.gen(x2, s, check_ctx), check_ctx),test.gen(y2, s, check_ctx), check_ctx);
		Expr body=check_ctx.mkEq(a, b);
		BoolExpr conjection = check_ctx.mkForall(bound,body,0,null,null,null,null);
		//System.out.println("check_gfg conjection: "+conjection);
		s.add(conjection);
		Status result = s.check();
		if(result == Status.SATISFIABLE) {
            //System.out.println("gfg sat");
			return true;
		}else if(result == Status.UNSATISFIABLE){
			//System.out.println("gfg unsat");
		return false;
		}else 
			//System.out.println("gfg unknow");
			return false;
	}
	public boolean check_commutative(Solver s,Context check_ctx,testz3 test) {
		s.reset();
		RealExpr x1=check_ctx.mkRealConst("x1");
        RealExpr x2=check_ctx.mkRealConst("x2");
        Expr left=test.agg(x1, x2, check_ctx);

        Expr right=test.agg(x2, x1, check_ctx);
        Expr[]bound=new Expr[]{x1,x2};
        Expr body = check_ctx.mkEq(left, right);
        BoolExpr conjection = check_ctx.mkForall(bound, body, 0, null ,null, null, null);
		//System.out.println("check_commutative conjection: "+conjection);
		s.add(conjection);
        Status result = s.check();
		if(result == Status.SATISFIABLE) {
            //System.out.println("commutative sat");
			return true;
		}else if(result == Status.UNSATISFIABLE){//System.out.println("commutative unsat");
		return false;
		}else 
			//System.out.println("commutative unknow");
			return false;
	}
	public boolean check_associative(Solver s,Context check_ctx,testz3 test) {
		s.reset();
		RealExpr x1=check_ctx.mkRealConst("x1");
        RealExpr x2=check_ctx.mkRealConst("x2");
        RealExpr x3=check_ctx.mkRealConst("x3");
		Expr[]bound=new Expr[]{x1,x2,x3};

        Expr left=test.agg(test.agg(x1, x2, check_ctx), x3, check_ctx);
        
        Expr right=test.agg(x1, test.agg(x2, x3, check_ctx), check_ctx);
        Expr body=check_ctx.mkEq(left, right);
        
        BoolExpr conjection = check_ctx.mkForall(bound,body,0,null,null,null,null);
		//System.out.println("associative conjection:"+conjection);
        s.add(conjection);
        Status result = s.check();
		if(result == Status.SATISFIABLE) {
            //System.out.println("associative sat");
			return true;
		}else if(result == Status.UNSATISFIABLE){//System.out.println("associative unsat");
		return false;
		}else 
			//System.out.println("associative unknow");
			return false;
	}
	public boolean check_agg_inverse(Solver s,Context check_ctx,testz3 test) {
		s.reset();
		Expr x1=check_ctx.mkRealConst("x1");
        Expr x2=check_ctx.mkRealConst("x2");
        Expr x3=check_ctx.mkRealConst("x3");
        Sort I=check_ctx.mkRealSort();
        FuncDecl f1=check_ctx.mkFuncDecl("f1",I,I);
		Expr x2_I=check_ctx.mkApp(f1, x2);
		
        Expr ari01=test.agg(x1, x3, check_ctx);
        Expr ari02=test.agg(x1, test.agg(test.agg(x2,x2_I, check_ctx),x3,check_ctx),check_ctx);
        
		Expr[] bound = new Expr[] {x1,x2,x3};
		Expr body = check_ctx.mkEq(ari01, ari02);
		BoolExpr conjection=check_ctx.mkForall(bound, body, 0, null, null, null, null);
        //System.out.println("agg_inverse conjection: "+conjection);
		s.add(conjection);
        //BoolExpr falseconjection = check_ctx.mkBool(conjection.isFalse());
        Status result = s.check();
		if(result == Status.SATISFIABLE) {
           // System.out.println("check_agg_inverse sat");
			return true;
		}else if(result == Status.UNSATISFIABLE){//System.out.println("check_agg_inverse unsat");
		return false;
		}
		return false;
	}
	public boolean check_singdep(Solver s,Context check_ctx,testz3 test) {
		s.reset();
		RealExpr x=check_ctx.mkRealConst("x");
        RealExpr y=check_ctx.mkRealConst("y");
        BoolExpr equation1=check_ctx.mkEq(test.agg(x, y, check_ctx), x);
        BoolExpr equation2=check_ctx.mkEq(test.agg(x, y, check_ctx), y);
        Expr body = check_ctx.mkOr(equation1,equation2);
        Expr[] bound=new Expr[]{x,y};
		BoolExpr conjection=check_ctx.mkForall(bound, body ,0, null, null, null, null);
		//System.out.println("singdep conjection: "+conjection);
		s.add(conjection);
        Status result = s.check();
		if(result == Status.SATISFIABLE) {
            //System.out.println("check_singdep sat");
			return true;
		}else if(result == Status.UNSATISFIABLE){//System.out.println("check_singdep unsat");
		return false;
		}else 
			//System.out.println("check_singdep unknow");
		return false;
	}

public static void main(String[] args) {
	try {
		//Scanner scanner = new Scanner(System.in);
		String application = null;
		String vfile = null;
		String efile = null;
		boolean use_cilk = true;
		double termcheck_threshold = 0.000001;
		int sssp_source = 0;
		int app_concurrency = 1;
		boolean directed = true;
		String out_prefix = null;
		for (int i = 0; i < args.length; i++) {
			if (args[i].equals("-application")) {
				application = args[++i];
			}

			if (args[i].equals("-vfile")) {
				vfile = args[++i];
			}

			if (args[i].equals("-efile")) {
				efile = args[++i];
			}
			if (args[i].equals("-directed")) {
				directed = Boolean.parseBoolean(args[++i]);
			}
			if (args[i].equals("-use_cilk")) {
				use_cilk = Boolean.parseBoolean(args[++i]);
			}
			if (args[i].equals("-termcheck_threshold")) {
				termcheck_threshold = Double.parseDouble(args[++i]);
			}
			if (args[i].equals("-app_concurrency")) {
				app_concurrency = Integer.parseInt(args[++i]);
			}
			if (args[i].equals("-sssp_source")) {
				sssp_source = Integer.parseInt(args[++i]);
			}
			if (args[i].equals("-out_prefix")) {
				out_prefix = args[++i];
			}
		}
		// String app_name = args[0];
		// System.out.println("运行的图算法： "+application); 
		testz3 test= new testz3();
		Context check_ctx=new Context();
		Solver s =check_ctx.mkSolver();
		Usage testUsage= new Usage();
		Choose choos=new Choose();
		//Engineer engineer = Engineer.MV;
		Engineer engineer = choos.choos_engine(s, test, check_ctx, testUsage);
		// System.out.println("engineer:"+engineer);
		try {
						System.out.println("mpirun -n 1 ../../../../../build/ingress -eng "+engineer+" -application "+application+" -vfile "+vfile+" -efile "+efile+" -directed="+directed+" -cilk="+use_cilk+" -termcheck_threshold "+termcheck_threshold+" -app_concurrency "+app_concurrency+" -sssp_source "+sssp_source+"  -out_prefix "+out_prefix+"");
            
						Process process2 = Runtime.getRuntime().exec( "mpirun -n 1 ../../../../../build/ingress -eng "+engineer+" -application "+application+" -vfile "+vfile+" -efile "+efile+" -directed="+directed+" -cilk="+use_cilk+" -termcheck_threshold "+termcheck_threshold+" -app_concurrency "+app_concurrency+" -sssp_source="+sssp_source+"  -out_prefix "+out_prefix);
            // System.out.println(process2.getInputStream());
						Reader reader = new InputStreamReader(process2.getErrorStream());
						BufferedReader br = new BufferedReader(reader);
						String line = null ;
						try{
							while((line = br.readLine() )!=null){
								System.out.println(line);
							}
						}catch(IOException e){
							e.printStackTrace();
						}
						int value = process2.waitFor();
						System.out.println(value);
					} catch (IOException e) {
            e.printStackTrace();
        }
		}catch(Exception e){}
	}
}

