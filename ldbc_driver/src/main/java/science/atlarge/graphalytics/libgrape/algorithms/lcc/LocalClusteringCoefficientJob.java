package science.atlarge.graphalytics.libgrape.algorithms.lcc;

import java.util.List;

import org.apache.commons.configuration.Configuration;

import science.atlarge.graphalytics.libgrape.LibgrapeJob;

public class LocalClusteringCoefficientJob extends LibgrapeJob {

	public LocalClusteringCoefficientJob(Configuration config, String graphName, String verticesPath, String edgesPath, boolean graphDirected, String jobId, String logPath) {
		super(config, graphName, verticesPath, edgesPath, graphDirected, jobId, logPath);
	}

	@Override
	protected void addJobArguments(List<String> args) {
		args.add("--application");
		args.add("lcc");
	}
}
