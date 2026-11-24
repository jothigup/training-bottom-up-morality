import unittest
from behavior_study.topic_policy import excluded_topics

class TopicPolicyTests(unittest.TestCase):
    def test_topics_in_actions(self):
        self.assertEqual(excluded_topics({'immoral': 'Calls them racist names.'}), ['racism'])
        self.assertEqual(excluded_topics({'prompt': 'They go to a strip club.'}), ['sex'])

    def test_ordinary_relationships_retained(self):
        self.assertEqual(excluded_topics({'prompt': 'She asks her boyfriend to lend her money.'}), [])

    def test_consequences_screened_when_present(self):
        self.assertEqual(excluded_topics({'moral_consequence': 'Reports racial discrimination.'}), ['racism'])

    def test_implicit_racial_reference(self):
        self.assertIn('racism', excluded_topics({'prompt':'He avoids black people.'}))

    def test_sexism_separate_from_sex(self):
        self.assertEqual(excluded_topics({'norm_for_review_only':'Sexist hiring is wrong.'}), ['sexism'])

    def test_race_in_review_dimension(self):
        self.assertIn('racism', excluded_topics({'review':{'ethical_dimension':'racism'}}))
