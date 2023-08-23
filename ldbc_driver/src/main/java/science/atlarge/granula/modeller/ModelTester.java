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
package science.atlarge.granula.modeller;

import science.atlarge.granula.archiver.GranulaArchiver;
import science.atlarge.granula.archiver.GranulaExecutor;
import science.atlarge.granula.modeller.entity.BasicType.ArchiveFormat;
import science.atlarge.granula.modeller.entity.Execution;
import science.atlarge.granula.modeller.job.JobModel;
import science.atlarge.granula.modeller.job.Overview;
import science.atlarge.granula.modeller.platform.Libgrape;
import science.atlarge.granula.modeller.source.JobDirectorySource;
import science.atlarge.granula.util.FileUtil;
import science.atlarge.granula.util.json.JsonUtil;

import java.nio.file.Paths;

/**
 * Created by wing on 21-8-15.
 */
public class ModelTester {
    public static void main(String[] args) {
        String inputPath = "/home/admin/Workstation/Exec/Granula/debug/archiver/libgrape/log";
        String outputPath = "/home/admin/Workstation/Exec/Granula/debug/archiver/libgrape/arc";

        Execution execution = (Execution) JsonUtil.fromJson(FileUtil.readFile(
                Paths.get(inputPath + "/execution/execution-log.js")), Execution.class);
        execution.setLogPath(inputPath);
        // Set end time in "log directory"/execution/execution-log.js, or the end time is set as the current time.
        execution.setEndTime(1489804044873l);
        execution.setArcPath(outputPath);
        JobModel jobModel = new JobModel(new Libgrape());

        GranulaExecutor granulaExecutor = new GranulaExecutor();
//        granulaExecutor.setEnvEnabled(false);
        granulaExecutor.setExecution(execution);
        granulaExecutor.buildJobArchive(jobModel);
    }
}
