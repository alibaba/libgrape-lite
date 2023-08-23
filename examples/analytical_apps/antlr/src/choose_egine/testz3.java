import com.microsoft.z3.ArithExpr;
import com.microsoft.z3.BoolExpr;
import com.microsoft.z3.Context;
import com.microsoft.z3.Expr;
import com.microsoft.z3.Solver;

public class testz3{














public Expr agg(Expr a,Expr b,Context check_gfg) { 
			Expr c = check_gfg.mkAdd(a,b);
			 return c;
} 
public Expr gen(Expr x2,Solver s,Context check_gfg) {
	Expr d = check_gfg.mkRealConst("0.85");
	Expr w = check_gfg.mkRealConst("w");
	Expr val = check_gfg.mkMul(x2,d,w);
	return val;
}																																																																																																																					 


}
