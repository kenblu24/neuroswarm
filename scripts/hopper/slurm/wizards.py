"""
Allow the user to choose parameters.
"""

from textual.app import App, ComposeResult, RenderResult
from textual.widget import Widget
from textual.widgets import Label, Input, Button
from textual.widgets import Collapsible
from textual.containers import Horizontal, HorizontalGroup, HorizontalScroll
from textual.containers import Vertical, VerticalGroup, VerticalScroll
from textual.widgets import RadioSet, RadioButton, SelectionList

# typing:
from typing import Any


class ParameterChooser(Widget):

    DEFAULT_CSS = """
    ParameterChooser {
        width: auto;
        layout: vertical;
    }
    Button.select_all, Button.deselect_all {
        width: auto;
        min-width: 1;
    }
    .align_right {
        align-horizontal: right;
    }
    .container.selection_buttons {
        margin-left: 1;
        margin-right: 1;
        width: 100%;
        dock: bottom;
    }
    Label.selection_label {
        margin-left: 1;
    }
    SelectionList {
        width: auto;
        min-width: 100w;
        max-width: 55;
        overflow: auto;
    }
    .option-list--option {
        width: auto;
        padding-right: 3;
    }
    """

    def __init__(self, choices: list[Any], label: str = '', **kwargs):
        super().__init__(**kwargs)
        self.label = label
        self.choices = choices
        self.choicetuples = [(str(c), c) for c in choices]
        self.selectionlist = SelectionList(*self.choicetuples)
        self.selectionlist.select_all()

    def compose(self) -> ComposeResult:
        yield Label(self.label, classes='selection_label')
        yield self.selectionlist
        with HorizontalGroup(classes='align_right container selection_buttons'):
            yield Button('✔', compact=True, classes='select_all')
            yield Button('❌', compact=True, classes='deselect_all')

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if 'deselect_all' in event.button.classes:
            self.selectionlist = self.selectionlist.deselect_all()
        elif 'select_all' in event.button.classes:
            self.selectionlist = self.selectionlist.select_all()

    def get_choices(self) -> list[Any]:
        return [c for c in self.choices if c in self.selectionlist.selected]


class RunParametrizer(App[dict[str, Any]]):
    CSS = """
    RunParametrizer {
        height: 100vh;
        overflow: hidden;
    }
    #advanced_grid {
        grid-columns: 1fr 1fr;
    }
    HorizontalScroll {
        width: 100%;
        height: 100fr;
    }
    ParameterChooser {
        height: 100%;
    }
    SelectionList {
        height: 100fr;
    }
    .footer {
        align-horizontal: right;
        background: #222;
    }
    """

    # def on_mount(self) -> None:  # Hover over elements to get a debugging tooltip.
    #     from textual_dominfo import DOMInfo
    #     DOMInfo.attach_to(self)

    def __init__(self, params: dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.params = params
        self.choosers = {k: ParameterChooser(choices=v, label=k)
                         for k, v in params.items()}

    def compose(self) -> ComposeResult:
        with HorizontalScroll(classes='main'):
            for chooser in self.choosers.values():
                yield chooser
        with Horizontal(classes='footer'):
            yield Button('Continue', id='return', compact=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == 'return':
            self.exit()

    def exit(self) -> None:
        super().exit(result=self.get_selections())

    def get_selections(self) -> dict[str, Any]:
        return {k: v.get_choices() for k, v in self.choosers.items()}


if __name__ == "__main__":
    params = {
        'eons_seed': [2026, 2027, 2028],
        'N': [6, 7, 8],
        'behavior': ['Circles', 'Aggregation', 'ExplodingDispersion', 'DelaunayDiffusion'],
        'rngstrat': ['TS1', 'TSG', 'TSR'],
    }
    app = RunParametrizer(params)
    app.run()
    print(app.get_selections())
