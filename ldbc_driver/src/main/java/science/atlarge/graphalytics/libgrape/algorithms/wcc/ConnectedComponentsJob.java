package science.atlarge.graphalytics.libgrape.algorithms.wcc;

import org.apache.commons.configuration.Configuration;
import science.atlarge.graphalytics.libgrape.LibgrapeJob;

import java.util.List;

public class ConnectedComponentsJob extends LibgrapeJob {

    public ConnectedComponentsJob(Configuration config, String graphName, String verticesPath, String edgesPath, boolean graphDirected, String jobId, String logPath) {
        super(config, graphName, verticesPath, edgesPath, graphDirected, jobId, logPath);
    }

    @Override
    protected void addJobArguments(List<String> args) {
        args.add("--application");
        args.add("wcc");
    }
}
