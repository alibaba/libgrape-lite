package science.atlarge.graphalytics.libgrape.algorithms.pr;

import org.apache.commons.configuration.Configuration;
import science.atlarge.graphalytics.domain.algorithms.PageRankParameters;
import science.atlarge.graphalytics.libgrape.LibgrapeJob;

import java.util.List;

public class PageRankJob extends LibgrapeJob {
    PageRankParameters params;

    public PageRankJob(Configuration config, String graphName, String verticesPath, String edgesPath,
                       boolean graphDirected, PageRankParameters params, String jobId, String logPath) {
        super(config, graphName, verticesPath, edgesPath, graphDirected, jobId, logPath);
        this.params = params;
    }

    @Override
    protected void addJobArguments(List<String> args) {
        args.add("--application");
        args.add("pagerank");
        args.add("--pr_d");
        args.add(Float.toString(params.getDampingFactor()));
        args.add("--pr_mr");
        args.add(Integer.toString(params.getNumberOfIterations()));
    }
}
