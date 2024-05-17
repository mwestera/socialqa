import click
import sys
import csv
import logging


@click.command(help="")
@click.argument("infile_rte",type=click.File())
@click.argument("infile_qa",type=click.File())
@click.argument("outfile",type=click.File(mode='w'), default=sys.stdout)
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

    logging.warning("Not to be used, see tODO.")

    with open(infile_rte) as f:
        tsv_reader_rte = csv.DictReader(infile_rte, delimiter='\t')
    with open(infile_qa) as f:
        tsv_reader_qa = csv.DictReader(infile_rte, delimiter='\t')
    with open(outfile, "w") as f:
        fieldnames = ['pivot_id', 'question', 'pivot', 'entailment', 'pivot_post', 'score_qa', 'score_rte']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for row_rte, row_qa in zip(tsv_reader_rte, tsv_reader_qa):
            # TODO: Shouldn't be a zip, but upon closer inspection I'm not sure triples make sense.
            #   Better just load from both files and do whichever pandas magic we need to do the analysis.
            new_row = {}
            if row_rte['pivot_id'] == row_qa['pivot_id']:
                new_row['pivot_id'] = row_rte['pivot_id']
                new_row['question'] = row_qa['question']
                new_row['pivot'] = row_rte['sentence1']
                new_row['entailment'] = row_rte['sentence2']
                new_row['pivot_post'] = row_qa['post']
                # new_row['score'] = scores
                new_row['score_qa'] = row_qa['score']
                new_row['score_rte'] = row_rte['score']
                writer.writerow(new_row)


if __name__ == '__main__':
    main()