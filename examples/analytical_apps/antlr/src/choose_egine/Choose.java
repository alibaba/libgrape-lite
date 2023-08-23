import com.microsoft.z3.BoolExpr;
import com.microsoft.z3.Context;
import com.microsoft.z3.Solver;
public class Choose {
    
    Engineer choos_engine(Solver s,testz3 test,Context check_ctx,Usage usage) {
		boolean is_gfg=usage.check_gfg_z3(s, check_ctx, test);
		//System.out.println("is_gfg:"+is_gfg);
		boolean is_commutative=usage.check_commutative(s, check_ctx, test);		
		//System.out.println("is_commutative: "+is_commutative);
		boolean is_associative=usage.check_associative(s, check_ctx, test);	
		//System.out.println("is_associative: "+is_associative);	
		boolean has_agg_inverse=usage.check_agg_inverse(s, check_ctx, test);	
		//System.out.println("has_agg_inverse:"+has_agg_inverse);
		
		boolean is_singdep=usage.check_singdep(s, check_ctx, test);
		//System.out.println("is_singdep: "+is_singdep);
		if(is_gfg&&is_commutative&&is_associative) {
			if(has_agg_inverse) {
				// System.out.println("Select MF engineer");
				return Engineer.MF;
			}else if (is_singdep) {
				// System.out.println("Use MP engineer");
				return Engineer.MP;
			}
		}
		if(has_agg_inverse) {
			// System.out.println("Use MV engineer");
			return Engineer.MV;
		}
		// System.out.println("Use ME engineer");
		return Engineer.ME;
	}
}
