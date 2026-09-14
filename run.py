"""Run prediction stages only; data generation is always a separate command."""

import argparse


def main():
    # Design item: Executed prediction conditions
    # Current setting: Pure-LLM and LLM-SimpleKT as main pipelines, plus gold-SimpleKT as a diagnostic condition.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=["extract", "llm-simplekt", "gold-simplekt", "pure-llm", "evaluate", "all"],
        nargs="?",
        default="all",
    )
    stage = parser.parse_args().stage

    if stage in ("extract", "all"):
        from experiment.llm_simplekt.extract import run
        run()
    if stage in ("llm-simplekt", "all"):
        from experiment.llm_simplekt.predict import run
        run("llm")
    if stage in ("gold-simplekt", "all"):
        from experiment.llm_simplekt.predict import run
        run("gold")
    if stage in ("pure-llm", "all"):
        from experiment.pure_llm.predict import run
        run()
    if stage in ("evaluate", "all"):
        from experiment.evaluation.evaluate import run
        run()


if __name__ == "__main__":
    main()
