       file n_installments  avg_length_chars  avg_length_words n_words
0  de.jsonl            118        6474.440678       1038.567797  122551
1  fr.jsonl             91        9233.538462       1610.175824  146526


We will cut them into 100 chunks per story.
Note: 1% isn’t 1% of the same story. Proportional chunks assume the translation is equally compressed everywhere.