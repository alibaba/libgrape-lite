package science.atlarge.graphalytics.libgrape;

import org.apache.commons.configuration.Configuration;
import org.apache.commons.exec.CommandLine;
import org.apache.commons.exec.DefaultExecutor;
import org.apache.commons.exec.Executor;
import org.apache.commons.exec.PumpStreamHandler;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.HashSet;

public abstract class LibgrapeJob {
    private static final Logger LOG = LogManager.getLogger(LibgrapeJob.class);

    private final String jobId;
    private final String graphName;
    private final String verticesPath;
    private final String edgesPath;
    private final boolean graphDirected;
    private File outputFile;
    private final Configuration config;
    private final String logPath;
    private long vertexNum;
    private long edgeNum;

    public LibgrapeJob(Configuration config, String graphName, String verticesPath, String edgesPath, boolean graphDirected, String jobId, String logPath) {
        this.config = config;
        this.graphName = graphName;
        this.verticesPath = verticesPath;
        this.edgesPath = edgesPath;
        this.graphDirected = graphDirected;
        this.jobId = jobId;
        this.logPath = logPath;
        this.vertexNum = -1;
        this.edgeNum = -1;
    }

    abstract protected void addJobArguments(List<String> args);

    public void setOutputFile(File file) {
        outputFile = file;
    }

    public void setVertexNum(long v) {
        vertexNum = v;
    }

    public void setEdgeNum(long e) {
        edgeNum = e;
    }

    public int run() throws IOException, InterruptedException {
        List<String> args = new ArrayList<>();
        args.add("--opt");
        args.add("--vfile");
        args.add(verticesPath);
        args.add("--efile");
        args.add(edgesPath);

        if (vertexNum != -1) {
            args.add("--vertex_num");
            args.add(String.valueOf(vertexNum));
        }
        if (edgeNum != -1) {
            args.add("--edge_num");
            args.add(String.valueOf(edgeNum));
        }

        String serialization_prefix = System.getenv("GRAPH_SERIALIZATION_DIR");
        if (serialization_prefix != null && !serialization_prefix.isEmpty()) {
            args.add("--deserialize");
            args.add("--serialization_prefix");
            serialization_prefix += "/";
            serialization_prefix += graphName;
            String nodes = config.getString("platform.libgrape.nodes");
            int nodesNum = nodes.split(",").length;
            serialization_prefix += "-";
            serialization_prefix += Integer.toString(nodesNum);
            args.add(serialization_prefix);
        }

        args.add(graphDirected ? "--directed" : "--nodirected");
        // args.add("--benchmarking");
        addJobArguments(args);

        if (outputFile != null) {
            args.add("--out_prefix");
            args.add(outputFile.getParentFile().getAbsolutePath());
        }

        String libgrapeHome = config.getString("platform.libgrape.home");
        String outputFilePath = outputFile.getAbsolutePath();

        int numThreads = config.getInt("platform.libgrape.num-threads", -1);

        if (numThreads > 0) {
            args.add("--ncpus");
            args.add(String.valueOf(numThreads));
        }

        args.add("--jobid");
        args.add(jobId);

        String argsString = "";

        for (String arg : args) {
            argsString += arg += " ";
        }

        String nodes = config.getString("platform.libgrape.nodes");
        String cmd = String.format("./bin/sh/run-mpi.sh %s %s %s %s %s %s", nodes, logPath, libgrapeHome,
                outputFilePath, LibgrapePlatform.LIBGRAPE_BINARY_NAME, argsString);

        LOG.info("executing command: " + cmd);

        CommandLine commandLine = CommandLine.parse(cmd);
        Executor executor = new DefaultExecutor();
        executor.setStreamHandler(new PumpStreamHandler(System.out, System.err));
        executor.setExitValue(0);
        return executor.execute(commandLine);

    /*
        ProcessBuilder pb = new ProcessBuilder(cmd.split(" "));
        pb.redirectErrorStream(true);

        Process process = pb.start();
        InputStreamReader isr = new InputStreamReader(process.getInputStream());
        BufferedReader br = new BufferedReader(isr);
        String line;
        while ((line = br.readLine()) != null) {
            System.out.println(line);
        }

        int exit = process.waitFor();

        if (exit != 0) {
            throw new IOException("unexpected error code");
        }
    */
    }
}
