package science.atlarge.graphalytics.libgrape.algorithms.sssp;

import org.apache.commons.configuration.Configuration;
import science.atlarge.graphalytics.domain.algorithms.SingleSourceShortestPathsParameters;
import science.atlarge.graphalytics.libgrape.LibgrapeJob;

import java.util.List;

public class SingleSourceShortestPathsJob extends LibgrapeJob {

    SingleSourceShortestPathsParameters params;

    public SingleSourceShortestPathsJob(Configuration config, String graphName, String verticesPath, String edgesPath, boolean graphDirected,
                                        SingleSourceShortestPathsParameters params, String jobId, String logPath) {
        super(config, graphName, verticesPath, edgesPath, graphDirected, jobId, logPath);
        this.params = params;
    }

    @Override
    protected void addJobArguments(List<String> args) {
        args.add("--application");
        args.add("sssp");
        args.add("--sssp_source");
        args.add(Long.toString(params.getSourceVertex()));
    }
}
