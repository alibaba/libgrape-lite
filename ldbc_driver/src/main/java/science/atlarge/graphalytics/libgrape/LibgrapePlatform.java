/*
 * Copyright 2015 Delft University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package science.atlarge.graphalytics.libgrape;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import science.atlarge.granula.archiver.PlatformArchive;
import science.atlarge.granula.modeller.job.JobModel;
import science.atlarge.granula.modeller.platform.Libgrape;
import science.atlarge.granula.util.FileUtil;
import org.apache.commons.io.output.TeeOutputStream;
import science.atlarge.graphalytics.configuration.ConfigurationUtil;
import science.atlarge.graphalytics.configuration.InvalidConfigurationException;
import science.atlarge.graphalytics.execution.BenchmarkRunSetup;
import science.atlarge.graphalytics.execution.RunSpecification;
import science.atlarge.graphalytics.execution.RuntimeSetup;
import science.atlarge.graphalytics.domain.graph.FormattedGraph;
import science.atlarge.graphalytics.domain.graph.LoadedGraph;
import science.atlarge.graphalytics.execution.BenchmarkRunner;
import science.atlarge.graphalytics.report.result.BenchmarkMetric;
import science.atlarge.graphalytics.report.result.BenchmarkMetrics;
import science.atlarge.graphalytics.report.result.BenchmarkRunResult;
import science.atlarge.graphalytics.domain.benchmark.BenchmarkRun;
import science.atlarge.graphalytics.granula.GranulaAwarePlatform;
import org.apache.commons.configuration.Configuration;
import org.apache.commons.configuration.PropertiesConfiguration;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import science.atlarge.graphalytics.execution.PlatformExecutionException;
import science.atlarge.graphalytics.domain.algorithms.BreadthFirstSearchParameters;
import science.atlarge.graphalytics.domain.algorithms.CommunityDetectionLPParameters;
import science.atlarge.graphalytics.domain.algorithms.PageRankParameters;
import science.atlarge.graphalytics.domain.algorithms.SingleSourceShortestPathsParameters;
import science.atlarge.graphalytics.libgrape.algorithms.bfs.BreadthFirstSearchJob;
import science.atlarge.graphalytics.libgrape.algorithms.cdlp.CommunityDetectionJob;
import science.atlarge.graphalytics.libgrape.algorithms.wcc.ConnectedComponentsJob;
import science.atlarge.graphalytics.libgrape.algorithms.pr.PageRankJob;
import science.atlarge.graphalytics.libgrape.algorithms.sssp.SingleSourceShortestPathsJob;
import science.atlarge.graphalytics.libgrape.algorithms.lcc.LocalClusteringCoefficientJob;
import org.json.simple.JSONObject;

/**
 * libgrape implementation of the Graphalytics benchmark.
 *
 * @author Stijn Heldens
 */
public class LibgrapePlatform implements GranulaAwarePlatform {
	protected static final Logger LOG = LogManager.getLogger();
	private static PrintStream sysOut;
	private static PrintStream sysErr;

	public static final String BENCHMARK_PROPERTIES_FILE = "benchmark.properties";
	private static final String GRANULA_PROPERTIES_FILE = "granula.properties";

	public static final String GRANULA_ENABLE_KEY = "benchmark.run.granula.enabled";
	public static String LIBGRAPE_BINARY_NAME = "bin/standard/run_app";

	private Configuration benchmarkConfig;


	public LibgrapePlatform() {

		Configuration granulaConfig;
		try {
			benchmarkConfig = ConfigurationUtil.loadConfiguration(BENCHMARK_PROPERTIES_FILE);
			granulaConfig = ConfigurationUtil.loadConfiguration(GRANULA_PROPERTIES_FILE);
		} catch(InvalidConfigurationException e) {
			LOG.warn("failed to load " + BENCHMARK_PROPERTIES_FILE, e);
			LOG.warn("Could not find or load \"{}\"", GRANULA_PROPERTIES_FILE);
			benchmarkConfig = new PropertiesConfiguration();
			granulaConfig = new PropertiesConfiguration();
		}

		boolean granulaEnabled = granulaConfig.getBoolean(GRANULA_ENABLE_KEY, false);
		LIBGRAPE_BINARY_NAME = granulaEnabled ? "./bin/granula/run_app": LIBGRAPE_BINARY_NAME;
	}

	@Override
	public void verifySetup() {

	}

	@Override
	public LoadedGraph loadGraph(FormattedGraph formattedGraph) throws Exception {
		return new LoadedGraph(formattedGraph, formattedGraph.getVertexFilePath(), formattedGraph.getEdgeFilePath());
	}

	@Override
	public void deleteGraph(LoadedGraph loadedGraph) {
		//
	}

	@Override
	public void prepare(RunSpecification runSpecification) {

	}

	@Override
	public void run(RunSpecification runSpecification) throws PlatformExecutionException {

		BenchmarkRun benchmarkRun = runSpecification.getBenchmarkRun();
		BenchmarkRunSetup benchmarkRunSetup = runSpecification.getBenchmarkRunSetup();
		RuntimeSetup runtimeSetup = runSpecification.getRuntimeSetup();

		LibgrapeJob job;
		Object params = benchmarkRun.getAlgorithmParameters();

		String logPath = benchmarkRunSetup.getLogDir().resolve("platform").toString();

		boolean graphDirected = benchmarkRun.getFormattedGraph().isDirected();
		String vertexFilePath = runtimeSetup.getLoadedGraph().getVertexPath();
		String edgeFilePath = runtimeSetup.getLoadedGraph().getEdgePath();

		switch(benchmarkRun.getAlgorithm()) {
			case BFS:
				job = new BreadthFirstSearchJob(benchmarkConfig, vertexFilePath, edgeFilePath,
						graphDirected, (BreadthFirstSearchParameters) params, benchmarkRun.getId(), logPath);
				break;
			case WCC:
				job = new ConnectedComponentsJob(benchmarkConfig, vertexFilePath, edgeFilePath,
						graphDirected, benchmarkRun.getId(), logPath);
				break;
			case LCC:
				job = new LocalClusteringCoefficientJob(benchmarkConfig, vertexFilePath, edgeFilePath,
						graphDirected, benchmarkRun.getId(), logPath);
				break;
			case CDLP:
				job = new CommunityDetectionJob(benchmarkConfig, vertexFilePath, edgeFilePath,
						graphDirected, (CommunityDetectionLPParameters) params, benchmarkRun.getId(), logPath);
				break;
			case PR:
				job = new PageRankJob(benchmarkConfig, vertexFilePath, edgeFilePath,
						graphDirected, (PageRankParameters) params, benchmarkRun.getId(), logPath);
				break;
			case SSSP:
				job = new SingleSourceShortestPathsJob(benchmarkConfig, vertexFilePath, edgeFilePath,
						graphDirected, (SingleSourceShortestPathsParameters) params, benchmarkRun.getId(), logPath);
				break;
			default:
				throw new PlatformExecutionException("Unsupported algorithm");
		}

		if (benchmarkRunSetup.isOutputRequired()) {
			Path outputFile = benchmarkRunSetup.getOutputDir().resolve(benchmarkRun.getName());
			job.setOutputFile(outputFile.toFile());
		}

		try {
			job.run();
		} catch (IOException|InterruptedException e) {
			throw new PlatformExecutionException("failed to execute command", e);
		}

	}

	@Override
	public void startup(RunSpecification runSpecification) {
		BenchmarkRunSetup benchmarkRunSetup = runSpecification.getBenchmarkRunSetup();
		startPlatformLogging(benchmarkRunSetup.getLogDir().resolve("platform").resolve("driver.logs"));
	}


	@Override
	public BenchmarkMetrics finalize(RunSpecification runSpecification) {
		stopPlatformLogging();
		BenchmarkRunSetup benchmarkRunSetup = runSpecification.getBenchmarkRunSetup();
		BenchmarkRun benchmarkRun = runSpecification.getBenchmarkRun();


		Path platformLogPath = benchmarkRunSetup.getLogDir().resolve("platform");

		final List<Double> superstepTimes = new ArrayList<>();

		try {
			Files.walkFileTree(platformLogPath, new SimpleFileVisitor<Path>() {
				@Override
				public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
					String logs = FileUtil.readFile(file);
					for (String line : logs.split("\n")) {
						if (line.contains("- run algorithm:")) {
							Pattern regex = Pattern.compile(
									".* - run algorithm: ([+-]?([0-9]*[.])?[0-9]+) sec.*");
							Matcher matcher = regex.matcher(line);
							matcher.find();
							superstepTimes.add(Double.parseDouble(matcher.group(1)));
						}
					}
					return FileVisitResult.CONTINUE;
				}
			});
		} catch (IOException e) {
			e.printStackTrace();
		}

		if (superstepTimes.size() != 0) {
			Double procTime = 0.0;
			for (Double superstepTime : superstepTimes) {
				procTime += superstepTime;
			}

			BenchmarkMetrics metrics = new BenchmarkMetrics();
			BigDecimal procTimeS = (new BigDecimal(procTime)).setScale(3, RoundingMode.CEILING);
			metrics.setProcessingTime(new BenchmarkMetric(procTimeS, "s"));

			return metrics;
		} else {
			LOG.error("Failed to find any metrics regarding superstep runtime.");
			return new BenchmarkMetrics();
		}
	}

	@Override
	public void enrichMetrics(BenchmarkRunResult benchmarkRunResult, Path arcDirectory) {
		try {
			PlatformArchive platformArchive = PlatformArchive.readArchive(arcDirectory);
			JSONObject processGraph = platformArchive.operation("ProcessGraph");
			BenchmarkMetrics metrics = benchmarkRunResult.getMetrics();

			Integer procTimeMS = Integer.parseInt(platformArchive.info(processGraph, "Duration"));
			BigDecimal procTimeS = (new BigDecimal(procTimeMS)).divide(new BigDecimal(1000), 3, BigDecimal.ROUND_CEILING);
			metrics.setProcessingTime(new BenchmarkMetric(procTimeS, "s"));

		} catch(Exception e) {
			LOG.error("Failed to enrich metrics.");
		}
	}

	@Override
	public void terminate(RunSpecification runSpecification) {
		BenchmarkRunner.terminatePlatform(runSpecification);
	}

	private static void startPlatformLogging(Path fileName) {
		sysOut = System.out;
		sysErr = System.err;
		try {
			File file = null;
			file = fileName.toFile();
			file.getParentFile().mkdirs();
			file.createNewFile();
			FileOutputStream fos = new FileOutputStream(file);
			TeeOutputStream bothStream =new TeeOutputStream(System.out, fos);
			PrintStream ps = new PrintStream(bothStream);
			System.setOut(ps);
			System.setErr(ps);
		} catch(Exception e) {
			e.printStackTrace();
			throw new IllegalArgumentException("cannot redirect to output file");
		}
		System.out.println("StartTime: " + System.currentTimeMillis());
	}

	private static void stopPlatformLogging() {
		System.out.println("EndTime: " + System.currentTimeMillis());
		System.setOut(sysOut);
		System.setErr(sysErr);
	}

	@Override
	public JobModel getJobModel() {
		return new JobModel(new Libgrape());
	}

	@Override
	public String getPlatformName() {
		return "libgrape";
	}

}
