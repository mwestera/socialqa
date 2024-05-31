import click
import sys
import json
import logging
import re
import string

import tqdm

import emoji

import difflib
htmldiff = difflib.HtmlDiff(wrapcolumn=80)


"""
To be run on collected, completed, anonymized posts, PRIOR to extract_sentences and make_tasks.

Removes URLs, removes punctuation-heavy lines, standardizes some quotation marks, and
optionally applies a spellchecker.

Example:

$ cat collected/posts.jsonl | python clean_posts.py --autocorrect hunspell --html autocorrect_viz.html > collected/posts_cleaned.jsonl

"""


PUNCTUATION_PROPORTION_THRESHOLD = .1   # removes lines with more punctuation than this
NUMERAL_PROPORTION_THRESHOLD = .2

@click.command(help="Extract sentences from posts and their context. Prints each sentence as a json dictionary, storing "
                    "the text along with various meta-info.")
@click.argument("posts", type=click.File('r'), default=sys.stdin)
@click.option("--autocorrect", help="Can choose between Hunspell (basic) and Neuspell (bert-based, slower); if omitted, no spellcheck is done.", type=click.Choice(['hunspell', 'neuspell'], case_sensitive=False), required=False, default=None)
@click.option("--html", help="Path to html file, will be used to vizualize spellcheck corrections.", type=click.Path(), required=False, default=None)
def main(posts, autocorrect, html):

    logging.basicConfig(level=logging.INFO)

    if html:
        with open(html, 'w') as file:
            file.write((htmldiff._file_template[:-40] % {'styles': difflib._styles, 'charset': 'utf-8'}) +
                       '\n' + htmldiff._legend)

    if autocorrect == 'hunspell':
        autocorrector = hunspell_autocorrect()
    elif autocorrect == 'neuspell':
        logging.warning('Neuspell autocorrect inevitably changes spacing, due to imperfect detokenization, '
                        'e.g., it doesn\'t currently glue apostrophes back on...')
        autocorrector = neuspell_autocorrect()
    else:
        logging.warning('Not doing autocorrect (use --autocorrect option if needed).')
        autocorrector = None

    try:
        for n_post, line in tqdm.tqdm(enumerate(posts), file=sys.stderr):
            post_entry = json.loads(line)
            clean_post_entry(post_entry, autocorrector, html=html)
            print(json.dumps(post_entry))
    except KeyboardInterrupt:
        logging.warning('Keyboard interrupt!')
        pass

    if html and autocorrect:
        with open(html, 'a') as file:
            file.write('\n\n</body></html>')
            logging.info(f'Html visualisation of spellchecker written to {html}')

    logging.info(f'Cleaned {n_post} post entries (inc. family).')


def neuspell_autocorrect():
    """
    Returns a function that takes a string and uses the Neuspell spellchecker
    and moses detokenizer for autocorrection.
    """
    import neuspell  # hmm, remember to manually install torch==1.13.1, transformers==4.30.2
    import sacremoses

    spellchecker = neuspell.BertChecker()
    spellchecker.from_pretrained()
    detokenizer = sacremoses.MosesDetokenizer('en').detokenize

    def correct_string(text):
        text = spellchecker.correct_string(text)
        text = detokenizer(text.split())
        return text

    return correct_string


def hunspell_autocorrect():
    """
    Returns a function that takes a string and uses the Hunspell spellchecker for autocorrection.
    """
    import hunspell

    spellchecker = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
    words_to_add = ['cybertruck',
                    'cybertrucks',
                    'koi',
                    'lol',
                    'ie',
                    'retransmitted',
                    'forecasted',
                    'ordnances',
                    ]    # Saw many more... so hunspell is kinda stupid?

    for word in words_to_add:
        spellchecker.add(word)
    word_regex = re.compile(r'\b(([A-Za-z]+\'?[A-Za-z]+)|([A-Za-z]+))\b')  # treat I'm, they're etc. as whole; don't check numerical things

    def correct_word(word):
        if not word[0].isupper() and not spellchecker.spell(word) and (suggestions := spellchecker.suggest(word)):
            return suggestions[0]
        return word

    def correct_string(text):
        corrected = word_regex.sub(lambda match: correct_word(match.group()), text)
        return corrected

    return correct_string


punctuation_translation = str.maketrans({
    '“': '"',       # quotation marks problematic for neuspell, perhaps for Bert tokenizer in general.
    '”': '"',
    '’': '\'',
    '‘': '\'',
    '…': '...',
})


md_url_regex = re.compile(r'\[([^\]]+)\]\(([^\)]+)\)')
empty_md_url_regex = re.compile(r'\[\]\(([^\)]+)\)')
# from  https://gist.github.com/gruber/8891611
plain_url_regex = re.compile(r'(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))')


def clean(text: str, spellchecker=None, html=None) -> str:
    """
    Returns a cleaned-up version of the text, optionally by using a provided spellchecker and,
    optionally-optionally, a detokenizer.

    Removes urls.
    Standardizes some non-ascii punctuation I encountered that Bert doesn't seem to like; there's probably more.
    Also removes punctuation-heavy lines.
    """
    text = text
    text = text.replace('[removed]', '')
    text = text.replace('[deleted]', '')
    text = emoji.replace_emoji(text, '')
    text = text.translate(punctuation_translation)
    text = md_url_regex.sub(r'\1', text)
    text = empty_md_url_regex.sub(r'<URL>', text)
    text = plain_url_regex.sub(r'<URL>', text)
    text = '\n'.join(line for line in text.split('\n') if line and proportion_punctuation(line) < PUNCTUATION_PROPORTION_THRESHOLD)
    text = '\n'.join(line for line in text.split('\n') if line and proportion_numerals(line) < NUMERAL_PROPORTION_THRESHOLD)
    if not (text := text.strip()):
        return text
    # TODO Maybe remove lines starting with |
    # TODO: remove things with too high proportion non-ascii (e.g., chinese).
    if spellchecker:
        text2 = spellchecker(text)
        if text != text2 and html:
            with open(html, 'a') as file:
                file.write(htmldiff.make_table(text.splitlines(), text2.splitlines()))
        text = text2
    return text.strip()


def proportion_punctuation(text):
    """
    Used in the main cleanup to remove overly punctuation-heavy lines.
    """
    return sum(1 for c in text if c in string.punctuation) / len(text)


def proportion_numerals(text):
    """
    Used in the main cleanup to remove overly punctuation-heavy lines.
    """
    return sum(c.isnumeric() for c in text) / len(text)



def clean_post_entry(post_entry: dict, spellchecker, html=None):
    """
    Iterates over all the posts (submission, parent, replies, post itself) of a .jsonl post entry,
    cleaning them all (modified in-place).
    """
    for post in iter_post_entry(post_entry):
        text_key = 'selftext' if post['type'] == 'submission' else 'body'
        old_text = post[text_key]
        clean_text = clean(old_text, spellchecker, html)
        post[text_key] = clean_text

        # for convenience, store under 'text' key as well, in which case, for submissions, include the title in that field.
        post['text'] = clean_text
        if post['type'] == 'submission':
            title = post['title'].strip()
            if post['title'][-1] not in string.punctuation:
                title += '.'
            post['text'] = title + '\n' + post['text']


def iter_post_entry(post_entry, already_done=None):
    """
    Iterates over a post's submission, parent, replies, etc., including the post itself;
    yielding each post only once.

    This is needed because lines in the .jsonl effectively contain multiple posts, all in need of cleanup.
    """
    if already_done is None:
        already_done = set()
    if post_entry['id'] in already_done:
        return

    already_done.add(post_entry['id'])
    yield post_entry

    if submission := post_entry.get('submission'):
        iter_post_entry(submission, already_done)
    if parent := post_entry.get('parent'):
        iter_post_entry(parent, already_done)
    for reply in post_entry['replies']:
        iter_post_entry(reply, already_done)


if __name__ == '__main__':
    main()
