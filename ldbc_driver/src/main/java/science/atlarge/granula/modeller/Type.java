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

import science.atlarge.granula.modeller.entity.BasicType;

public class Type extends BasicType {

    // actor
    public static String libgrape = "libgrape";

    // mission
    public static String Job = "Job";
    public static String LoadGraph = "LoadGraph";
    public static String OffloadGraph = "OffloadGraph";
    public static String ProcessGraph = "ProcessGraph";

    // info
    public static String StartTime = "StartTime";
    public static String EndTime = "EndTime";
    public static String Bsp = "Bsp";
    public static String Superstep = "Superstep";

    public static String Cleanup = "Cleanup";
    public static String Startup = "Startup";

}
