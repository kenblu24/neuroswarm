"""
An App to show the current time.
"""

from datetime import datetime

from textual.app import App, ComposeResult
from textual.widgets import Label, Input
from textual.widgets import Collapsible
from textual.containers import Horizontal, HorizontalGroup, HorizontalScroll
from textual.containers import Vertical, VerticalGroup, VerticalScroll
from textual.widgets import RadioSet, RadioButton, SelectionList


class SlurmWizard(App):
    CSS = """
    #advanced_grid {
        grid-columns: 1fr 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        with Collapsible(title="Common Slurm Options", collapsed=False):
            with HorizontalGroup():
                yield Label("Job Name: ")
                yield Input(placeholder="default", compact=True)
            with HorizontalGroup():
                yield Label("Time Limit: ")
                yield Input(placeholder="default", compact=True)
                yield Label("Memory: ")
                yield Input(placeholder="4G", compact=True, max_length=8)
                yield Label("CPUs: ")
                yield Input(placeholder="auto", compact=True)
        with Collapsible(title="Advanced Slurm Options", collapsed=True):
            with HorizontalGroup():
                with VerticalGroup():
                    yield Label("Partition:")
                    with RadioSet(name="partition", id="partition", compact=True):
                        yield RadioButton("normal (3 days)", value=True)
                        yield RadioButton("debug (1 hr)")
                        yield RadioButton("interactive (12 hrs)")
                        yield RadioButton("contrib (6 days)")
                        yield RadioButton("gpuq (1 day)")
                        yield RadioButton("other (custom)")
                    yield Input(placeholder="Custom partition name", compact=True, disabled=True)
                with VerticalGroup():
                    yield Label("Email:")
                    yield SelectionList[str](
                        ("BEGIN", "BEGIN", True),
                        ("END", "END", True),
                        ("FAIL", "FAIL", True),
                        ("REQUEUE", "REQUEUE", False),
                        compact=True,
                    )



    # def on_ready(self) -> None:
    #     self.update_clock()
    #     self.set_interval(1, self.update_clock)

    # def update_clock(self) -> None:
    #     clock = datetime.now().time()
    #     self.query_one(Digits).update(f"{clock:%T}")


if __name__ == "__main__":
    app = SlurmWizard()
    app.run()
