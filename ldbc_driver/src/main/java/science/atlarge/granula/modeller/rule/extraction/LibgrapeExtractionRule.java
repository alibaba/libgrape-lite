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
package science.atlarge.granula.modeller.rule.extraction;

import science.atlarge.granula.modeller.rule.extraction.ExtractionRule;
import science.atlarge.granula.modeller.source.DataStream;
import science.atlarge.granula.modeller.source.log.Log;
import science.atlarge.granula.modeller.source.log.LogLocation;
import science.atlarge.granula.util.UuidGenerator;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by wing on 21-8-15.
 */
public class LibgrapeExtractionRule extends ExtractionRule {

    public LibgrapeExtractionRule(int level) {
        super(level);
    }

    @Override
    public boolean execute() {
        return false;
    }

    public String generateText(String infoName, String infoValue, String actorType, String actorId, String missionType,
                               String missionId, String operationUuid) {
        String text = String.format("GRANULA - InfoName:%s InfoValue:%s ActorType:%s ActorId:%s MissionType:%s MissionId:%s RecordUuid:%s OperationUuid:%s Timestamp:%s\n",
                infoName, infoValue, actorType, actorId, missionType,
                missionId, UuidGenerator.getRandomUUID(), operationUuid, System.currentTimeMillis());
        return text;
    }

    public List<Log> extractLogFromInputStream(DataStream dataStream) {

        List<Log> granularlogList = new ArrayList<>();

        try {
            BufferedReader br = new BufferedReader(
                    new InputStreamReader(dataStream.getInputStream()));

            String line = null;
            int lineCount = 0;

            while ((line = br.readLine()) != null) {
                lineCount++;

                if(line.contains("GRANULA") ) {

                    if(line.contains("libgrape") && line.contains("Job")) {
                        continue;
                    }

                    Log log = extractRecord(line);

                    LogLocation trace = new LogLocation();

                    String codeLocation;
                    String logFilePath;
                    if(false) { //TODO if supported
                        codeLocation = line.split("\\) - Granular")[0].split(" \\(")[1];
                    }

                    codeLocation = "unspecified";
                    logFilePath = "unspecified";

                    trace.setLocation(logFilePath, lineCount, codeLocation);
                    log.setLocation(trace);

                    granularlogList.add(log);
                }
            }

            br.close();


        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return granularlogList;
    }



    public Log extractRecord(String line) {
        Log log = new Log();

        String granularLog = line.split("GRANULA ")[1];
        String[] logAttrs = granularLog.split("\\s+");

        for (String logAttr : logAttrs) {
            if (logAttr.contains(":")) {
                String[] attrKeyValue = logAttr.split(":");
                if (attrKeyValue.length == 2) {

                    String name = attrKeyValue[0];
                    String value = attrKeyValue[1];
                    String unescapedValue = value.replaceAll("\\[COLON\\]", ":").replaceAll("\\[SPACE\\]", " ");

                    log.addLogInfo(name, unescapedValue);
                } else {
                    log.addLogInfo(attrKeyValue[0], "");
                }
            }
        }
        return log;
    }
}
