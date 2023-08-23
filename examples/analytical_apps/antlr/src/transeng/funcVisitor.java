import java.util.List;
import java.util.ListIterator;
import java.util.Map;

import org.antlr.v4.misc.OrderedHashMap;
import org.antlr.v4.runtime.Parser;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.ParseTreeListener;
import org.antlr.v4.runtime.tree.TerminalNode;
@SuppressWarnings("unchecked")
public class funcVisitor extends CPP14ParserBaseVisitor<Object> {
    public static String agg_Str;
    public static String gen_Str;

    @Override public Object visitFunctionDefinition(CPP14Parser.FunctionDefinitionContext ctx)
    { String func = null;
        // System.out.println("--------------------------------------------------------------------------------------------");
        StringBuffer sb = new StringBuffer();

        func=ctx.getText();
		int aggCheck=sb.append(func).indexOf("boolaccumulate");
        int genCheck=sb.append(func).indexOf("generate");
        if(aggCheck!=-1)
        {
            agg_Str=func;
            // System.out.println(func);
        }
        else if(genCheck!=-1)
        {
            gen_Str=func;
            // System.out.println(func);
        }
        return visitChildren(ctx);
    }


}
