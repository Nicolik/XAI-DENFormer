import subprocess
from pathlib import Path

import paths

# ==============================
# CONFIG
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCRIPTS_DIR = Path(__file__).resolve().parent / 'scripts'
REPORTS_DIR = Path(__file__).resolve().parent / 'reports'
TRAIN_SCRIPT = SCRIPTS_DIR / '01_run_train.py'
INFERENCE_SCRIPT = SCRIPTS_DIR / '02_run_inference.py'
INFERENCE_ATTN_SCRIPT = SCRIPTS_DIR / '03_run_inference_attn.py'
ATTN_AGGREGATE_SCRIPT = SCRIPTS_DIR / '04_run_attn_aggregate.py'
ATTN_BOX_SCRIPT = SCRIPTS_DIR / '05_run_attn_box.py'
ATTN_STAT_PANEL_SCRIPT = SCRIPTS_DIR / '06_run_attn_stat_panel.py'
ATTN_STRATEGY_CONSENSUS_SCRIPT = SCRIPTS_DIR / '07_run_attn_strategy_consensus.py'
GXI_SCRIPT = SCRIPTS_DIR / '11_run_gxi.py'
GXI_AGGREGATE_SCRIPT = SCRIPTS_DIR / '12_run_gxi_aggregate.py'
GXI_BOX_SCRIPT = SCRIPTS_DIR / '13_run_gxi_box.py'
GXI_STAT_PANEL_SCRIPT = SCRIPTS_DIR / '14_run_gxi_stat_panel.py'
GXI_STRATEGY_CONSENSUS_SCRIPT = SCRIPTS_DIR / '15_run_gxi_strategy_consensus.py'
EMBEDDINGS_SCRIPT = SCRIPTS_DIR / '08_run_embeddings.py'
XAI_EMBEDDINGS_SCRIPT = SCRIPTS_DIR / '09_run_xai_embeddings.py'
XAI_OVERALL_PANEL_SCRIPT = SCRIPTS_DIR / '16_run_xai_overall_panel.py'
CONFMAT_REPORT_SCRIPT = REPORTS_DIR / 'make_confmat.py'
METRICS_REPORT_SCRIPT = REPORTS_DIR / 'make_metrics.py'
MODEL_TRADEOFF_REPORT_SCRIPT = REPORTS_DIR / 'make_model_tradeoff.py'
SPLITS_DIR = paths.splits_dir
SPLIT_FILES = paths.split_files

SPLIT_TODO = {
    'continent': True,
    'timebin': True,
    'cdhit': True,
}

# ==============================
# PIPELINE FLAGS
# ==============================
RUN_TRAIN = False
RUN_INFERENCE = False
RUN_INFERENCE_ATTN = False

RUN_ATTN_AGGREGATE = False
RUN_ATTN_BOX = False
RUN_ATTN_STAT_PANEL = False
RUN_ATTN_STRATEGY_CONSENSUS = False

RUN_GXI = False
RUN_GXI_AGGREGATE = False
RUN_GXI_BOX = False
RUN_GXI_STAT_PANEL = False
RUN_GXI_STRATEGY_CONSENSUS = False

RUN_EMBEDDINGS = False
RUN_XAI_EMBEDDINGS = False
RUN_XAI_OVERALL_PANEL = False

RUN_CONFMAT_REPORT = False
RUN_METRICS_REPORT = False
RUN_MODEL_TRADEOFF_REPORT = False

REPORT_SUMMARY = 'median'
REPORT_ERROR_BAR = 'minmax'
REPORT_RESULT_TABLE_PERCENT = True

# ==============================
# EXPERIMENT CONFIG
# ==============================
EXPERIMENTS = [
    # {'model_type': 'logreg', 'pooling': 'mean'},
    # {'model_type': 'ffnn', 'pooling': 'mean'},
    # {'model_type': 'denformer', 'pooling': 'first'},
    # {'model_type': 'denformer', 'pooling': 'mean'},
    # {'model_type': 'denformer', 'pooling': 'max'},
    # {'model_type': 'performer', 'pooling': 'mean'},
    # {'model_type': 'longformer', 'pooling': 'mean'},
]

ATTN_EXPERIMENTS = [
    {'model_type': 'denformer_attn', 'pooling': 'first'},
    {'model_type': 'denformer_attn', 'pooling': 'mean'},
    {'model_type': 'denformer_attn', 'pooling': 'max'},
]

GXI_EXPERIMENTS = [
    {'model_type': 'denformer_attn', 'pooling': 'first'},
    {'model_type': 'denformer_attn', 'pooling': 'mean'},
    {'model_type': 'denformer_attn', 'pooling': 'max'},
]

XAI_EMBEDDING_EXPERIMENTS = [
    {'model_type': 'denformer', 'pooling': 'first'},
    {'model_type': 'denformer', 'pooling': 'mean'},
    {'model_type': 'denformer', 'pooling': 'max'},
]

EPOCHS = 100
K = 1
ONE_HOT = True

FOLD = None


# ==============================
# UTILS
# ==============================
def run_command(cmd):
    print('\n' + '=' * 80)
    print('RUNNING:')
    print(' '.join(cmd))
    print('=' * 80 + '\n')

    process = subprocess.Popen(cmd)
    process.wait()

    if process.returncode != 0:
        raise RuntimeError(f'Command failed with code {process.returncode}')


def build_common_args(script_path):
    cmd = [
        'python',
        str(script_path),
        '--epochs', str(EPOCHS),
        '--k', str(K),
    ]

    if ONE_HOT:
        cmd.append('--one-hot')

    if FOLD is not None:
        cmd.extend(['--fold', str(FOLD)])

    return cmd


def build_step_command(script_path, split_path, run_name, model_type, pooling):
    cmd = build_common_args(script_path)
    cmd.extend([
        '--split_file', str(split_path),
        '--run_name', run_name,
        '--model_type', model_type,
        '--pooling', pooling,
    ])
    return cmd






def build_metrics_report_command(script_path):
    cmd = [
        'python',
        str(script_path),
        '--summary', REPORT_SUMMARY,
        '--error-bar', REPORT_ERROR_BAR,
    ]
    if REPORT_RESULT_TABLE_PERCENT:
        cmd.append('--result-table-percent')
    return cmd


def build_model_tradeoff_report_command(script_path):
    return [
        'python',
        str(script_path),
        '--summary', REPORT_SUMMARY,
        '--error-bar', REPORT_ERROR_BAR,
    ]


# ==============================
# MAIN
# ==============================
def main():
    print(f'PROJECT_ROOT: {PROJECT_ROOT}')

    for name, filename in SPLIT_FILES.items():
        if not SPLIT_TODO.get(name, False):
            print(f'\n>>> Skipping split strategy: {name}')
            continue

        split_path = SPLITS_DIR / filename
        if not split_path.exists():
            print(f'WARNING: split file not found, skipping: {split_path}')
            continue

        # ==============================
        # NEURAL MODELS
        # ==============================
        for exp in EXPERIMENTS:
            model_type = exp['model_type']
            pooling = exp['pooling']

            print('\n' + '#' * 80)
            print(f'### SPLIT: {name} | MODEL: {model_type} | POOLING: {pooling}')
            print('#' * 80)

            if RUN_TRAIN:
                print(f'\n>>> Running TRAIN for: {name} | model={model_type} | pooling={pooling}')
                train_cmd = build_step_command(TRAIN_SCRIPT, split_path, name, model_type, pooling)
                run_command(train_cmd)

            if RUN_INFERENCE:
                print(f'\n>>> Running INFERENCE for: {name} | model={model_type} | pooling={pooling}')
                inference_cmd = build_step_command(INFERENCE_SCRIPT, split_path, name, model_type, pooling)
                run_command(inference_cmd)

        # ==============================
        # ATTENTION / XAI ATTENTION (optional)
        #
        # These steps are intentionally independent:
        # - RUN_INFERENCE_ATTN regenerates fold-level attention outputs.
        # - RUN_ATTN_AGGREGATE reuses existing attention outputs and builds aggregate plots/tables.
        # - RUN_ATTN_BOX reuses existing attention outputs and builds box/statistical plots/tables,
        #   including dataset-level output-box-dataset results.
        # - RUN_ATTN_STAT_PANEL reuses output-box-dataset and builds paper-level composite
        #   panels for raw p-values and raw effect sizes.
        #
        # This makes it possible to rerun only 05_run_attn_box.py without rerunning
        # the expensive 03_run_inference_attn.py step.
        # ==============================
        if RUN_INFERENCE_ATTN or RUN_ATTN_AGGREGATE or RUN_ATTN_BOX or RUN_ATTN_STAT_PANEL:
            for exp in ATTN_EXPERIMENTS:
                model_type = exp['model_type']
                pooling = exp['pooling']

                print('\n' + '#' * 80)
                print(f'### ATTN | SPLIT: {name} | MODEL: {model_type} | POOLING: {pooling}')
                print('#' * 80)

                if RUN_INFERENCE_ATTN:
                    print(f'\n>>> Running INFERENCE ATTN for: {name}')
                    inference_attn_cmd = build_step_command(
                        INFERENCE_ATTN_SCRIPT, split_path, name, model_type, pooling
                    )
                    run_command(inference_attn_cmd)

                if RUN_ATTN_AGGREGATE:
                    print(f'\n>>> Running ATTN AGGREGATE for: {name}')
                    attn_aggregate_cmd = build_step_command(
                        ATTN_AGGREGATE_SCRIPT, split_path, name, model_type, pooling
                    )
                    run_command(attn_aggregate_cmd)

                if RUN_ATTN_BOX:
                    print(f'\n>>> Running ATTN BOX for: {name}')
                    attn_box_cmd = build_step_command(
                        ATTN_BOX_SCRIPT, split_path, name, model_type, pooling
                    )
                    run_command(attn_box_cmd)

                if RUN_ATTN_STAT_PANEL:
                    print(f'\n>>> Running ATTN STAT PANEL for: {name}')
                    attn_stat_panel_cmd = build_step_command(
                        ATTN_STAT_PANEL_SCRIPT, split_path, name, model_type, pooling
                    )
                    run_command(attn_stat_panel_cmd)

        # ==============================
        # GRADIENT x INPUT (GxI) XAI (optional)
        #
        # These steps mirror attention aggregation but use input gradients.
        # GxI has its own experiment block so it can be controlled independently
        # from ATTN_EXPERIMENTS while keeping the same default models/poolings.
        # - RUN_GXI computes fold-level per-sample GxI .npy profiles.
        # - RUN_GXI_AGGREGATE reuses existing GxI outputs and builds aggregate plots.
        # - RUN_GXI_BOX reuses existing GxI outputs and builds region-level box/statistical tables.
        # - RUN_GXI_STAT_PANEL reuses gxi_box/output-box-dataset and builds per-split composite panels.
        # - RUN_GXI_STRATEGY_CONSENSUS reuses all three gxi_box dataset outputs and builds the consensus panel.
        # ==============================
        if RUN_GXI or RUN_GXI_AGGREGATE or RUN_GXI_BOX or RUN_GXI_STAT_PANEL or RUN_XAI_OVERALL_PANEL:
            for exp in GXI_EXPERIMENTS:
                model_type = exp['model_type']
                pooling = exp['pooling']

                print('\n' + '#' * 80)
                print(f'### GxI | SPLIT: {name} | MODEL: {model_type} | POOLING: {pooling}')
                print('#' * 80)

                if RUN_GXI:
                    print(f'\n>>> Running GxI for: {name}')
                    gxi_cmd = build_step_command(
                        GXI_SCRIPT, split_path, name, model_type, pooling
                    )
                    run_command(gxi_cmd)

                if RUN_GXI_AGGREGATE:
                    print(f'\n>>> Running GxI AGGREGATE for: {name}')
                    gxi_aggregate_cmd = build_step_command(
                        GXI_AGGREGATE_SCRIPT, split_path, name, model_type, pooling
                    )
                    run_command(gxi_aggregate_cmd)

                if RUN_GXI_BOX:
                    print(f'\n>>> Running GxI BOX for: {name}')
                    gxi_box_cmd = build_step_command(
                        GXI_BOX_SCRIPT, split_path, name, model_type, pooling
                    )
                    run_command(gxi_box_cmd)

                if RUN_GXI_STAT_PANEL:
                    print(f'\n>>> Running GxI STAT PANEL for: {name}')
                    gxi_stat_panel_cmd = build_step_command(
                        GXI_STAT_PANEL_SCRIPT, split_path, name, model_type, pooling
                    )
                    run_command(gxi_stat_panel_cmd)

                if RUN_XAI_OVERALL_PANEL:
                    print(f'\n>>> Running OVERALL XAI PANEL for: {name}')
                    xai_panel_cmd = build_step_command(
                        XAI_OVERALL_PANEL_SCRIPT, split_path, name, model_type, pooling
                    )
                    run_command(xai_panel_cmd)

        # ==============================
        # EMBEDDING XAI (optional)
        # ==============================
        if RUN_EMBEDDINGS or RUN_XAI_EMBEDDINGS:
            for exp in XAI_EMBEDDING_EXPERIMENTS:
                model_type = exp['model_type']
                pooling = exp['pooling']

                if RUN_EMBEDDINGS:
                    print(f'\n>>> Running EMBEDDINGS for: {name} | model={model_type} | pooling={pooling}')
                    embeddings_cmd = build_step_command(
                        EMBEDDINGS_SCRIPT, split_path, name, model_type, pooling
                    )
                    run_command(embeddings_cmd)

                if RUN_XAI_EMBEDDINGS:
                    print(f'\n>>> Running XAI EMBEDDING PLOTS for: {name} | model={model_type} | pooling={pooling}')
                    xai_embeddings_cmd = build_step_command(
                        XAI_EMBEDDINGS_SCRIPT, split_path, name, model_type, pooling
                    )
                    run_command(xai_embeddings_cmd)

    # ==============================
    # PAPER REPORTS (optional)
    # ==============================
    if RUN_CONFMAT_REPORT:
        print('\n>>> Running confusion-matrix report')
        run_command(['python', str(CONFMAT_REPORT_SCRIPT)])

    if RUN_METRICS_REPORT:
        print('\n>>> Running metrics report')
        run_command(build_metrics_report_command(METRICS_REPORT_SCRIPT))

    if RUN_MODEL_TRADEOFF_REPORT:
        print('\n>>> Running model trade-off report')
        run_command(build_model_tradeoff_report_command(MODEL_TRADEOFF_REPORT_SCRIPT))

    if RUN_ATTN_STRATEGY_CONSENSUS:
        for exp in ATTN_EXPERIMENTS:
            pooling = exp['pooling']
            print('\n' + '#' * 80)
            print(f'### ATTN STRATEGY CONSENSUS | POOLING: {pooling}')
            print('#' * 80)
            attn_consensus_cmd = build_common_args(ATTN_STRATEGY_CONSENSUS_SCRIPT)
            attn_consensus_cmd.extend([
                '--pooling', pooling,
            ])
            run_command(attn_consensus_cmd)

    if RUN_GXI_STRATEGY_CONSENSUS:
        for exp in GXI_EXPERIMENTS:
            pooling = exp['pooling']
            print('\n' + '#' * 80)
            print(f'### GxI STRATEGY CONSENSUS | POOLING: {pooling}')
            print('#' * 80)
            gxi_consensus_cmd = build_common_args(GXI_STRATEGY_CONSENSUS_SCRIPT)
            gxi_consensus_cmd.extend([
                '--pooling', pooling,
            ])
            run_command(gxi_consensus_cmd)

    print('\nWorkflow completed.')


if __name__ == '__main__':
    main()
