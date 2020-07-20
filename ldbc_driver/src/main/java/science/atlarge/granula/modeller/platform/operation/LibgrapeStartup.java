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

package science.atlarge.granula.modeller.platform.operation;

import science.atlarge.granula.modeller.Type;
import science.atlarge.granula.modeller.rule.derivation.ColorDerivation;
import science.atlarge.granula.modeller.rule.derivation.FilialCompletenessDerivation;
import science.atlarge.granula.modeller.rule.derivation.SimpleSummaryDerivation;
import science.atlarge.granula.modeller.rule.derivation.time.*;
import science.atlarge.granula.modeller.rule.linking.UniqueParentLinking;

public class LibgrapeStartup extends AbstractOperationModel {

    public LibgrapeStartup() {
        super(Type.libgrape, Type.Startup);
    }

    public void loadRules() {
        super.loadRules();
        addLinkingRule(new UniqueParentLinking(Type.libgrape, Type.Job));

        addInfoDerivation(new JobStartTimeDerivation(1));
        addInfoDerivation(new SiblingEndTimeDerivation(5, Type.libgrape, Type.LoadGraph));
        addInfoDerivation(new DurationDerivation(6));
        this.addInfoDerivation(new FilialCompletenessDerivation(2));

        String summary = "Prepare.";
        addInfoDerivation(new SimpleSummaryDerivation(11, summary));

        addInfoDerivation(new ColorDerivation(11, "#666"));
    }

}

