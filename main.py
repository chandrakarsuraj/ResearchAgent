
from research_agent.pipelines.pipelines import build_pipeline 
from rich import print

def main():
    result = build_pipeline("how indian students are performing in the field of AI and ML")
    print(result)


if __name__ == "__main__":
    main()