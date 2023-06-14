package science.atlarge.graphalytics.libgrape;

import org.apache.commons.configuration.Configuration;
import org.apache.commons.configuration.PropertiesConfiguration;
import org.apache.commons.io.output.TeeOutputStream;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import science.atlarge.graphalytics.configuration.ConfigurationUtil;
import science.atlarge.graphalytics.configuration.InvalidConfigurationException;
import science.atlarge.graphalytics.domain.algorithms.*;
import science.atlarge.graphalytics.domain.benchmark.BenchmarkRun;
import science.atlarge.graphalytics.domain.graph.FormattedGraph;
import science.atlarge.graphalytics.domain.graph.LoadedGraph;
import science.atlarge.graphalytics.execution.*;
import science.atlarge.graphalytics.libgrape.algorithms.bfs.BreadthFirstSearchJob;
import science.atlarge.graphalytics.libgrape.algorithms.cdlp.CommunityDetectionJob;
import science.atlarge.graphalytics.libgrape.algorithms.lcc.LocalClusteringCoefficientJob;
import science.atlarge.graphalytics.libgrape.algorithms.pr.PageRankJob;
import science.atlarge.graphalytics.libgrape.algorithms.sssp.SingleSourceShortestPathsJob;
import science.atlarge.graphalytics.libgrape.algorithms.wcc.ConnectedComponentsJob;
import science.atlarge.graphalytics.report.result.BenchmarkMetric;
import science.atlarge.graphalytics.report.result.BenchmarkMetrics;

import java.io.*;
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

public class LibgrapePlatform implements Platform {
    public static final String BENCHMARK_PROPERTIES_FILE = "benchmark.properties";
    public static final String GPU_ENABLE_KEY = "platform.run.gpu.enabled";
    protected static final Logger LOG = LogManager.getLogger();
    public static String LIBGRAPE_BINARY_NAME = "bin/standard/run_app";
    public static String LIBGRAPE_GPU_BINARY_NAME = "bin/standard/run_cuda_app";
    private static PrintStream sysOut;
    private static PrintStream sysErr;
    private Configuration benchmarkConfig;

    public LibgrapePlatform() {
        try {
            benchmarkConfig = ConfigurationUtil.loadConfiguration(BENCHMARK_PROPERTIES_FILE);
        } catch (InvalidConfigurationException e) {
            LOG.warn("failed to load " + BENCHMARK_PROPERTIES_FILE, e);
            benchmarkConfig = new PropertiesConfiguration();
            boolean gpuEnabled = benchmarkConfig.getBoolean(GPU_ENABLE_KEY, false);
            LIBGRAPE_BINARY_NAME =  gpuEnabled ? LIBGRAPE_GPU_BINARY_NAME : LIBGRAPE_BINARY_NAME;
        }
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
            TeeOutputStream bothStream = new TeeOutputStream(System.out, fos);
            PrintStream ps = new PrintStream(bothStream);
            System.setOut(ps);
            System.setErr(ps);
        } catch (Exception e) {
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
    public void verifySetup() {

    }

    @Override
    public LoadedGraph loadGraph(FormattedGraph formattedGraph) throws Exception {
        return new LoadedGraph(formattedGraph, formattedGraph.getVertexFilePath(), formattedGraph.getEdgeFilePath());
    }

    @Override
    public void prepare(RunSpecification runSpecification) {

    }

    @Override
    public void deleteGraph(LoadedGraph loadedGraph) {
        //
    }

    @Override
    public void run(RunSpecification runSpecification) throws PlatformExecutionException {

        BenchmarkRun benchmarkRun = runSpecification.getBenchmarkRun();
        BenchmarkRunSetup benchmarkRunSetup = runSpecification.getBenchmarkRunSetup();
        RuntimeSetup runtimeSetup = runSpecification.getRuntimeSetup();

        Algorithm algorithm = benchmarkRun.getAlgorithm();
        boolean graphDirected = benchmarkRun.getFormattedGraph().isDirected();
        String vertexFilePath = runtimeSetup.getLoadedGraph().getVertexPath();
        String edgeFilePath = runtimeSetup.getLoadedGraph().getEdgePath();

        Object params = benchmarkRun.getAlgorithmParameters();

        String logPath = benchmarkRunSetup.getLogDir().resolve("platform").toString();

        long vertexNum = benchmarkRun.getGraph().getNumberOfVertices();
        long edgeNum = benchmarkRun.getGraph().getNumberOfEdges();

        String graphName = benchmarkRun.getGraph().getName();

        LibgrapeJob job;
        switch (algorithm) {
            case BFS:
                job = new BreadthFirstSearchJob(benchmarkConfig, graphName, vertexFilePath, edgeFilePath,
                        graphDirected, (BreadthFirstSearchParameters) params, benchmarkRun.getId(), logPath);
                break;
            case WCC:
                job = new ConnectedComponentsJob(benchmarkConfig, graphName, vertexFilePath, edgeFilePath,
                        graphDirected, benchmarkRun.getId(), logPath);
                break;
            case LCC:
                job = new LocalClusteringCoefficientJob(benchmarkConfig, graphName, vertexFilePath, edgeFilePath,
                        graphDirected, benchmarkRun.getId(), logPath);
                break;
            case CDLP:
                job = new CommunityDetectionJob(benchmarkConfig, graphName, vertexFilePath, edgeFilePath,
                        graphDirected, (CommunityDetectionLPParameters) params, benchmarkRun.getId(), logPath);
                break;
            case PR:
                job = new PageRankJob(benchmarkConfig, graphName, vertexFilePath, edgeFilePath,
                        graphDirected, (PageRankParameters) params, benchmarkRun.getId(), logPath);
                break;
            case SSSP:
                job = new SingleSourceShortestPathsJob(benchmarkConfig, graphName, vertexFilePath, edgeFilePath,
                        graphDirected, (SingleSourceShortestPathsParameters) params, benchmarkRun.getId(), logPath);
                break;
            default:
                throw new PlatformExecutionException("Unsupported algorithm");
        }

        job.setVertexNum(vertexNum);
        job.setEdgeNum(edgeNum);

        if (benchmarkRunSetup.isOutputRequired()) {
            Path outputFile = benchmarkRunSetup.getOutputDir().resolve(benchmarkRun.getName());
            job.setOutputFile(outputFile.toFile());
        }

        try {
            int exitCode = job.run();
            if (exitCode != 0) {
                throw new PlatformExecutionException("Libgrape exited with an error code: " + exitCode);
            }
        } catch (IOException | InterruptedException e) {
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
                    try (BufferedReader reader = new BufferedReader(new FileReader(file.toFile()))) {
                        String line;
                        while ((line = reader.readLine()) != null) {
                            try {
                                if (line.contains("- run algorithm:")) {
                                    Pattern regex = Pattern.compile(".* - run algorithm: ([+-]?([0-9]*[.])?[0-9]+) sec.*");
                                    Matcher matcher = regex.matcher(line);
                                    matcher.find();
                                    superstepTimes.add(Double.parseDouble(matcher.group(1)));
                                }
                            } catch (Exception e) {
                                e.printStackTrace();
                            }
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
    public void terminate(RunSpecification runSpecification) {
        BenchmarkRunner.terminatePlatform(runSpecification);
    }

    @Override
    public String getPlatformName() {
        return "libgrape";
    }
}
