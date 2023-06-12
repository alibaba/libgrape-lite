package science.atlarge.graphalytics.libgrape.algorithms.cdlp;

import org.apache.commons.configuration.Configuration;
import science.atlarge.graphalytics.domain.algorithms.CommunityDetectionLPParameters;
import science.atlarge.graphalytics.libgrape.LibgrapeJob;

import java.util.List;

public class CommunityDetectionJob extends LibgrapeJob {
    private final CommunityDetectionLPParameters params;

    public CommunityDetectionJob(Configuration config, String graphName, String verticesPath, String edgesPath, boolean graphDirected,
                                 CommunityDetectionLPParameters params, String jobId, String logPath) {
        super(config, graphName, verticesPath, edgesPath, graphDirected, jobId, logPath);
        this.params = params;
    }

    @Override
    protected void addJobArguments(List<String> args) {
        args.add("--application");
        args.add("cdlp");
        args.add("--cdlp_mr");
        args.add(Integer.toString(params.getMaxIterations()));
    }
}
