import logging
import click
from thunder.api.handlers import app
from thunder.engine.train import train_chunk
import pandas as pd

logger = logging.getLogger(__name__)

@click.group()
def cli():
    """Thunder ML Pipeline Management CLI"""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to run the API server")
@click.option("--port", default=8000, help="Port to run the API server")
def serve(host, port):
    """Start the FastAPI inference server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port, proxy_headers=True)


@cli.command()
@click.option(
    "--input",
    required=True,
    type=click.Path(exists=True),
    help="Path to input data file",
)
@click.option("--size", default=500, help="Chunk size")
def chunk(input, size):
    """Run the preprocessing pipeline"""
    import itertools

    chunks = pd.read_csv(input, chunksize=size)
    chunks_tee, chunks_iter = itertools.tee(chunks)
    chunks_iter = iter(chunks_tee)
    checkpoint_path = None
    preprocessor_path = None


    chunk_counter = 1
    for chunk_df in chunks_iter:
        next_chunk = next(chunks_iter, None)
        is_last = next_chunk is None

        logging.warning("Processing chunk: %s", chunk_counter)
        checkpoint_path, preprocessor_path = train_chunk(
            chunk_df,
            checkpoint_path=checkpoint_path,
            preprocessor_path=preprocessor_path,
            last=is_last,
        )
        chunk_counter += 1


if __name__ == "__main__":
    cli()
