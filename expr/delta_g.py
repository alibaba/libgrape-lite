#!/usr/bin/env python3
import sys
from subprocess import getstatusoutput
import re
import csv

graphbolt_prefix = "/mnt/data/nfs/graphbolt"
tornado_prefix = "/mnt/data/nfs/AutoInc/build"
dataset_prefix = "/mnt/data/nfs/dataset"
serial_prefix = "/mnt/data/nfs/ser"

# change portion of base file
percentage = "0.10"
run_times = 2
workers_num = "16"


def get_dataset():
    dataset = {'road_usa': 16651563,
               'europe_osm': 44499591,
               'uk-2005': 23311901,
               'twitter': 1037947}
    dataset = {'uk-2005': 23311901}
    dynamic_dataset = {'dewiki.JAN': 5588,
                       'dewiki.FEB': 5588,
                       'dewiki.MAR': 5588,
                       'dewiki.APR': 5588,
                       'dewiki.MAY': 5588}

    if len(sys.argv) == 3 and sys.argv[2] == "dynamic":
        return dynamic_dataset
    return dataset


def run(cmd):
    status, out = getstatusoutput(cmd)
    with open('/tmp/expr.log', 'a') as fo:
        fo.write(cmd + "\n")
        fo.write(out + "\n")
    if status != 0:
        print("Failed to execute " + cmd)
    return status, out


def get_prefix(name: str):
    if "dewiki" in name:
        return dataset_prefix + "/wiki-de"
    else:
        return "{PREFIX}/{NAME}/{PERCENTAGE}".format(PREFIX=dataset_prefix, NAME=name, PERCENTAGE=percentage)


def get_base(name: str, adj: bool, weighted: bool, directed: bool):
    if adj:
        ext = ".adj"
    else:
        ext = ".base"
    extra = ""
    if weighted:
        extra = "_w"
    if not directed:
        extra = "_ud"
    return "{PREFIX}/{NAME}{EXTRA}{EXT}".format(PREFIX=get_prefix(name), NAME=name,
                                                EXTRA=extra,
                                                EXT=ext)


def get_update(name: str, weighted: bool, directed: bool):
    ext = ".update"
    extra = ""
    if weighted:
        extra = "_w"
    if not directed:
        extra = "_ud"

    return "{PREFIX}/{NAME}{EXTRA}{EXT}".format(PREFIX=get_prefix(name), NAME=name,
                                                EXTRA=extra,
                                                EXT=ext)


def get_updated(name: str, weighted: bool, directed: bool):
    ext = ".updated"
    extra = ""
    if weighted:
        extra = "_w"
    if not directed:
        extra = "_ud"

    return "{PREFIX}/{NAME}{EXTRA}{EXT}".format(PREFIX=get_prefix(name), NAME=name,
                                                EXTRA=extra,
                                                EXT=ext)


def get_vfile(name: str):
    if "dewiki" in name:
        return "{PREFIX}/wiki-de/{NAME}.v".format(PREFIX=dataset_prefix, NAME=name)
    else:
        return "{PREFIX}/{NAME}/{NAME}_w.v".format(PREFIX=dataset_prefix, PERCENTAGE=percentage, NAME=name)


def run_graphbolt():
    dataset = get_dataset()
    fields = [name for name in dataset]
    fields.insert(0, "")

    for app in ('PageRank', 'SSSP'):
        table = [["Batch(s)"], ["Inc(s)"], ["Mem(MB)"]]
        with open("graphbolt_" + app + percentage + ".csv", 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            for name, source in dataset.items():
                ok = True
                weighted = False
                directed = True
                if app == 'PHP' or app == 'SSSP':
                    weighted = True
                elif app == 'CC':
                    directed = False

                base_file = get_base(name, True, weighted, directed)
                update_file = get_update(name, weighted, directed)
                cmd = "wc -l %s" % update_file
                status, out = run(cmd)
                if status != 0:
                    sys.exit("Failed to execute wc")
                batch_size = out.split()[0]
                init_time_sum = 0
                inc_time_sum = 0
                mem_sum = 0

                cmd = "LD_PRELOAD={GB_PREFIX}/lib/mimalloc/out/release/libmimalloc.so " \
                      "{GB_PREFIX}/apps/{APP} " \
                      "-numberOfUpdateBatches 1 -nEdges {BATCH_SIZE} " \
                      "-maxIters 18 " \
                      "-streamPath {UPDATE} " \
                      "-source={SOURCE} " \
                      "-nWorkers {WORKERS_NUM} " \
                      "{BASE}".format(GB_PREFIX=graphbolt_prefix, APP=app,
                                      BATCH_SIZE=batch_size,
                                      UPDATE=update_file,
                                      BASE=base_file,
                                      SOURCE=source,
                                      WORKERS_NUM=workers_num)

                for curr_round in range(run_times):
                    print("Evaluating({ROUND}) {APP} on {NAME}".format(ROUND=curr_round, APP=app, NAME=name))
                    status, gb_out = run(cmd)
                    if status != 0:
                        ok = False
                        break

                    for line in gb_out.split('\n'):
                        match = re.match("^Initial graph processing : (.*?)$", line)
                        if match:
                            init_time_sum += float(match.groups()[0])
                        match = re.match("^Finished batch : (.*?)$", line)
                        if match:
                            inc_time_sum += float(match.groups()[0])
                        match = re.match("^Mem: (.*?) MB$", line)
                        if match:
                            mem_sum += float(match.groups()[0])
                batch_time = "Failed"
                inc_time = "Failed"
                mem = "Failed"
                if ok:
                    batch_time = str(init_time_sum / run_times)
                    inc_time = str(inc_time_sum / run_times)
                    mem = str(str(mem_sum / run_times))

                table[0].append(batch_time)
                table[1].append(inc_time)
                table[2].append(mem)

                print("Avg Init: " + batch_time)
                print("Avg Inc: " + inc_time)
                print("Mem: " + mem)
            csvwriter.writerows(table)


def run_tornado():
    dataset = get_dataset()
    fields = [name for name in dataset]
    fields.insert(0, "")

    for app in ('tornado_pagerank',):
        table = [["Batch(s)"], ["Inc(s)"], ["Mem(MB)"]]
        with open(app + percentage + ".csv", 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            for name, source in dataset.items():
                ok = True
                weighted = False
                if app == "tornado_php":
                    weighted = True
                vfile = get_vfile(name)
                efile = get_base(name, False, weighted, True)
                efile_updated = get_updated(name, weighted, True)
                init_time_sum = 0
                inc_time_sum = 0
                mem_sum = 0

                cmd = "mpirun -n 1 {BIN_PREFIX}/{APP} " \
                      "-vfile {VFILE} " \
                      "-efile {EFILE} " \
                      "-efile_updated {efile_updated} " \
                      "-directed " \
                      "-app_concurrency {WORKERS_NUM} " \
                      "-pr_d 0.85 -pr_tol 0.01 " \
                      "-serialization_prefix {SERIAL_PREFIX} " \
                      "-php_d 0.8 -php_tol 0.01 -php_source {SOURCE}".format(
                    WORKERS_NUM=workers_num,
                    BIN_PREFIX=tornado_prefix,
                    APP=app,
                    VFILE=vfile,
                    EFILE=efile,
                    efile_updated=efile_updated,
                    SERIAL_PREFIX=serial_prefix,
                    SOURCE=source)

                for curr_round in range(run_times):
                    print("Evaluating({ROUND}) {APP} on {NAME}".format(ROUND=curr_round, APP=app, NAME=name))
                    status, tornado_out = run(cmd)

                    if status != 0:
                        ok = False
                        break

                    for line in tornado_out.split('\n'):
                        match = re.match("^.*?Query: 0: (.*?) sec$", line)
                        if match:
                            init_time_sum += float(match.groups()[0])
                        match = re.match("^.*?Query: 1: (.*?) sec$", line)
                        if match:
                            inc_time_sum += float(match.groups()[0])
                        match = re.match("^.*?Mem: (.*?) MB$", line)
                        if match:
                            mem_sum += float(match.groups()[0])

                batch_time = "Failed"
                inc_time = "Failed"
                mem = "Failed"
                if ok:
                    batch_time = str(init_time_sum / run_times)
                    inc_time = str(inc_time_sum / run_times)
                    mem = str(str(mem_sum / run_times))

                table[0].append(batch_time)
                table[1].append(inc_time)
                table[2].append(mem)

                print("Avg Init: " + batch_time)
                print("Avg Inc: " + inc_time)
                print("Mem: " + mem)
            csvwriter.writerows(table)


def run_autoinc(rerun: bool):
    dataset = get_dataset()
    fields = [name for name in dataset]
    fields.insert(0, "")

    for app in ('pagerank', 'sssp', 'gcn'):
        if rerun:
            table = [["Batch(s)"], ["Mem(MB)"]]
        else:
            table = [["Batch(s)"], ["Inc(s)"], ["Mem(MB)"]]
        out_name = "autoinc_" + app + percentage + ".csv"
        if rerun:
            out_name = "autoinc_rerun_" + app + percentage + ".csv"
        with open(out_name, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            for name, source in dataset.items():
                ok = True
                weighted = False
                directed = True
                cilk = "false"
                if app in ("php", "sssp"):
                    weighted = True
                elif app == "cc":
                    directed = False
                if dataset in ('road_usa', 'europe_osm'):
                    cilk = "true"
                vfile = get_vfile(name)
                efile = get_base(name, False, weighted, directed)
                efile_update = get_update(name, weighted, directed)
                efile_updated = get_updated(name, weighted, directed)
                init_time_sum = 0
                inc_time_sum = 0
                mem_sum = 0

                if rerun:
                    cmd = "mpirun -n 1 {BIN_PREFIX}/ingress " \
                          "-application {APP} " \
                          "-vfile {VFILE} " \
                          "-efile {EFILE} " \
                          "-directed " \
                          "-cilk={CILK} " \
                          "-segmented_partition=false " \
                          "-termcheck_threshold 0.01 " \
                          "-app_concurrency {WORKERS_NUM} " \
                          "-serialization_prefix {SERIAL_PREFIX} " \
                          "-sssp_source {SOURCE} -php_source {SOURCE}".format(
                        BIN_PREFIX=tornado_prefix,
                        CILK=cilk,
                        APP=app,
                        TYPE=type,
                        VFILE=vfile,
                        EFILE=efile_updated,
                        WORKERS_NUM=workers_num,
                        SERIAL_PREFIX=serial_prefix,
                        SOURCE=source)
                else:
                    cmd = "mpirun -n 1 {BIN_PREFIX}/ingress " \
                          "-application {APP} " \
                          "-vfile {VFILE} " \
                          "-efile {EFILE} " \
                          "-efile_update {EFILE_UPDATE} " \
                          "-efile_updated {EFILE_UPDATED} " \
                          "-directed " \
                          "-cilk={CILK} " \
                          "-segmented_partition=false " \
                          "-termcheck_threshold 0.01 " \
                          "-app_concurrency {WORKERS_NUM} " \
                          "-serialization_prefix {SERIAL_PREFIX} " \
                          "-sssp_source {SOURCE} -php_source {SOURCE}".format(
                        BIN_PREFIX=tornado_prefix,
                        CILK=cilk,
                        APP=app,
                        TYPE=type,
                        VFILE=vfile,
                        EFILE=efile,
                        EFILE_UPDATE=efile_update,
                        EFILE_UPDATED=efile_updated,
                        WORKERS_NUM=workers_num,
                        SERIAL_PREFIX=serial_prefix,
                        SOURCE=source)

                for curr_round in range(run_times):
                    print("Evaluating({ROUND}) {APP} on {NAME}".format(ROUND=curr_round, APP=app, NAME=name))
                    status, autoinc_out = run(cmd)

                    if status != 0:
                        ok = False
                        break

                    for line in autoinc_out.split('\n'):
                        match = re.match("^.*?Batch time: (.*?) sec$", line)
                        if match:
                            init_time_sum += float(match.groups()[0])

                        match = re.match("^.*?Inc time: (.*?) sec$", line)
                        if match:
                            inc_time_sum += float(match.groups()[0])
                        match = re.match("^.*?Mem: (.*?) MB$", line)
                        if match:
                            mem_sum += float(match.groups()[0])
                batch_time = "Failed"
                inc_time = "Failed"
                mem = "Failed"
                if ok:
                    batch_time = str(init_time_sum / run_times)
                    inc_time = str(inc_time_sum / run_times)
                    mem = str(str(mem_sum / run_times))

                table[0].append(batch_time)
                if not rerun:
                    table[1].append(inc_time)
                    table[2].append(mem)
                else:
                    table[1].append(mem)

                print("Avg Init: " + batch_time)
                if not rerun:
                    print("Avg Inc: " + inc_time)
                print("Mem: " + mem)
            csvwriter.writerows(table)


if __name__ == '__main__':
    for _ in ("0.05", "0.10", "0.15", "0.20"):
        percentage = _
        print(percentage)
        if sys.argv[1] == 'graphbolt':
            run_graphbolt()
        elif sys.argv[1] == 'tornado':
            run_tornado()
        elif sys.argv[1] == 'autoinc':
            run_autoinc(False)
        elif sys.argv[1] == 'autoinc_rerun':
            run_autoinc(True)
        else:
            sys.exit("Invalid arg: " + sys.argv[1])
