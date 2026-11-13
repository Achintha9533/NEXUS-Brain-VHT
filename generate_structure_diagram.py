import os
from graphviz import Digraph

os.makedirs("output", exist_ok=True)

dot = Digraph(comment="test_main Architecture", format="png")
dot.attr(rankdir="TB", size="8,10", fontname="Helvetica")
dot.attr('node', fontname="Helvetica", style="filled,rounded", margin="0.2")

dot.node("Setup", "1. Test Class Setup & Mocking\n• TestMainOrchestration (unittest.TestCase)\n• Patches run_pipeline, HBM fit, & forecast", fillcolor="#E1F5FE", shape="box")
dot.node("Mocks", "2. Mock Data Initialization\n• Mock patient objects & clinical DataFrame\n• Mock volume history & summary results DataFrame", fillcolor="#FFF9C4", shape="box")
dot.node("Exec", "3. Temporary Directory Execution\n• Switches working directory to temp folder\n• Invokes main.main() orchestration controller", fillcolor="#FFE0B2", shape="box")
dot.node("Assert", "4. Verification & Assertions\n• Asserts mock functions called once\n• Verifies output/run_metadata.json exists", fillcolor="#C8E6C9", shape="box")

dot.edge("Setup", "Mocks", label=" prepares test environment")
dot.edge("Mocks", "Exec", label=" injects mock data")
dot.edge("Exec", "Assert", label=" validates execution")

output_path = "output/test_main_structure"
dot.render(output_path, cleanup=True)
print(f"Successfully generated test_main structure diagram at: {output_path}.png")