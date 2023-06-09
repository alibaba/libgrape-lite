package science.atlarge.graphalytics.libgrape.algorithms.bfs;

import org.apache.commons.configuration.Configuration;
import science.atlarge.graphalytics.domain.algorithms.BreadthFirstSearchParameters;
import science.atlarge.graphalytics.libgrape.LibgrapeJob;

import java.util.List;

public class BreadthFirstSearchJob extends LibgrapeJob {

    BreadthFirstSearchParameters params;

    public BreadthFirstSearchJob(Configuration config, String graphName, String verticesPath, String edgesPath, boolean graphDirected,
                                 BreadthFirstSearchParameters params, String jobId, String logPath) {
        super(config, graphName, verticesPath, edgesPath, graphDirected, jobId, logPath);
        this.params = params;
    }

    @Override
    protected void addJobArguments(List<String> args) {
        args.add("--application");
        args.add("bfs");
        args.add("--bfs_source");
        args.add(Long.toString(params.getSourceVertex()));
    }
}
