import typer
from ucr_benchmark_template import dataset
from ucr_benchmark_template.modeling import train

app = typer.Typer()

app.add_typer(dataset.app, name="dataset")
app.add_typer(train.app, name="train")

if __name__ == "__main__":
    app()