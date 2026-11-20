import json
import unittest
from pathlib import Path

from llmsim.runner import simulate
from llmsim.scenario import load_scenario_dict, scenario_from_dict
from llmsim.viz import render_html

SCENARIO = Path(__file__).resolve().parent.parent / "configs" / "scenarios" / "deepseek_v41_s3.toml"


class HtmlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = render_html(simulate(scenario_from_dict(load_scenario_dict(SCENARIO))))

    def test_parts_are_assembled(self):
        for placeholder in ("/*STYLE*/", "/*SCRIPT*/", "__DATA__"):
            self.assertNotIn(placeholder, self.html)

    def test_embeds_every_layer_and_card(self):
        data = json.loads(self.html.split("const D = ", 1)[1].split(";\n", 1)[0])
        self.assertEqual([len(ph["layers"]) for ph in data["phases"]], [40, 40])
        self.assertEqual(len(data["phases"][0]["layers"][0]["dispatch"]["sent"]), data["ep"])


if __name__ == "__main__":
    unittest.main()
