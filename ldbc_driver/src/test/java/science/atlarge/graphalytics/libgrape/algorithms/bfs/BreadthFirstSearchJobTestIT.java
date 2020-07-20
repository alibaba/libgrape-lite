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
package science.atlarge.graphalytics.libgrape.algorithms.bfs;

import java.io.File;

import science.atlarge.graphalytics.domain.algorithms.BreadthFirstSearchParameters;
import science.atlarge.graphalytics.libgrape.Utils;
import science.atlarge.graphalytics.validation.GraphStructure;
import science.atlarge.graphalytics.validation.algorithms.bfs.BreadthFirstSearchOutput;
import science.atlarge.graphalytics.validation.algorithms.bfs.BreadthFirstSearchValidationTest;

/**
 * Validation tests for the BFS implementation in libgrape.
 *
 * @author Stijn Heldens
 */
public class BreadthFirstSearchJobTestIT extends BreadthFirstSearchValidationTest {

	@Override
	public BreadthFirstSearchOutput executeDirectedBreadthFirstSearch(GraphStructure graph,
			BreadthFirstSearchParameters parameters) throws Exception {
		return execute(graph, parameters, true);
	}

	@Override
	public BreadthFirstSearchOutput executeUndirectedBreadthFirstSearch(GraphStructure graph,
			BreadthFirstSearchParameters parameters) throws Exception {
		return execute(graph, parameters, false);
	}
	
	private BreadthFirstSearchOutput execute(GraphStructure graph,
			BreadthFirstSearchParameters parameters, boolean directed) throws Exception {
		File edgesFile = File.createTempFile("edges.", ".txt");
		File verticesFile = File.createTempFile("vertices.", ".txt");
		File outputFile = File.createTempFile("output.", ".txt");

		Utils.writeEdgeToFile(graph, directed, edgesFile);
		Utils.writeVerticesToFile(graph, verticesFile);

		String jobId = "RandomJobId";
		String logPath = "RandomLogDir";

		BreadthFirstSearchJob job = new BreadthFirstSearchJob(
				Utils.loadConfiguration(),
				verticesFile.getAbsolutePath(), edgesFile.getAbsolutePath(),
				directed, parameters, jobId, logPath);
		job.setOutputFile(outputFile);
		job.run();
		
		return new BreadthFirstSearchOutput(Utils.readResults(outputFile, Long.class));
	}
}
