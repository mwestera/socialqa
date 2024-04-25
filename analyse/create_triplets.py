import click
import sys
import csv

@click.command(help="")
@click.argument("infile_rte",type=str, default=sys.stdin)
@click.argument("infile_qa",type=str, default=sys.stdin)
@click.argument("outfile",type=str, default=sys.stdout)


def main(infile_rte, infile_qa, outfile):
    """
    Process the input files and create triplets based on matching pivot IDs.

    Args:
        infile_rte (str): Path to the RTE input file.
        infile_qa (str): Path to the QA input file.
        outfile (str): Path to the output file.

    Returns:
        None
    """
    with open(infile_rte) as f:
        tsv_reader_rte = csv.DictReader(infile_rte, delimiter='\t')
    with open(infile_qa) as f:
        tsv_reader_qa = csv.DictReader(infile_rte, delimiter='\t')
    with open(outfile, "w") as f:
        fieldnames = ['pivot_id', 'question', 'pivot', 'entailment', 'pivot_post', 'score']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for row_rte, row_qa in zip(tsv_reader_rte, tsv_reader_qa):
            new_row = {}
            if row_rte['pivot_id'] == row_qa['pivot_id']:
                scores = (row_rte['score'], row_qa['score'])
                new_row['pivot_id'] = row_rte['pivot_id']
                new_row['pivot_post'] = row_qa['post']
                new_row['pivot'] = row_rte['sentence1']
                new_row['question'] = row_qa['question']
                new_row['entailment'] = row_rte['sentence2']
                row_rte['score'] = scores
                writer.writerow(row_rte)
