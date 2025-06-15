from openai import OpenAI
import os
import torch
import json

from src.dependencies.config import EMBEDDING_MODEL

example1 = """
Background of the Merger

    The terms and conditions of the merger agreement and merger are the result
of arm's length negotiations between representatives of Chordiant and
representatives of Prime Response. Set forth below is a summary of the
background of these negotiations.

    Chordiant and Prime Response have been familiar with each other's businesses
for a number of years. Senior executives of the two companies have previously
encountered one another in a variety of business and industry settings.

    Throughout the summer and early fall of 2000, Prime Response engaged in
discussions with a number of companies concerning potential business
combinations or other partnering transactions.

    On May 23, 2000, Peter Boni, Prime Response's President and Chief Executive
Officer, met with representatives of Thomas Weisel Partners, to explore
strategic partnership options, including a possible marketing partnership with
some of Thomas Weisel Partners' clients.

    On June 21, 2000, Samuel Spadafora, Chordiant's Chief Executive Officer and
Chairman of the Chordiant board of directors called Mr. Boni to explore
strategic partnership options, including a possible marketing partnership.

    During July 2000, Thomas Weisel Partners discussed with Chordiant various
topics related to a possible transaction with Prime Response. The engagement of
Thomas Weisel Partners was confirmed in writing by an engagement letter dated
October 24, 2000. On July 12, 2000, Thomas Weisel Partners gave Mr. Spadafora,
Steven Springsteel, Chordiant's then current Chief Financial Officer, Don
Morrison, Executive Vice President, Business Development and Marketing, David
Bernstein, Vice President of Strategic Business Development and

Joseph Tumminaro, Chief Technology Officer of Chordiant, a briefing on Prime
Response's financial and product background.
"""

example2 = """
Background of the Merger

        During the last several years, Mediconsult has held conversations with
a number of companies to evaluate possible business combinations or strategic
alliances. In the third quarter of 2000, the Mediconsult board of directors
began considering alternatives to maximize shareholder value and raise
additional capital.

        Mediconsult's advisors contacted a number of companies that they and
Mediconsult had identified as candidates for a possible business combination or
strategic alliance with Mediconsult. In the last week of August, 2000, Robert A.
Jennings, the then Chairman of Mediconsult, received an indication of interest
from a third party regarding a possible strategic transaction or business
combination between the two companies.

        Beginning the first week in September 2000, Ian Sutcliffe, Chief
Executive Officer of Mediconsult, and E. Michael Ingram, Chief Financial Officer
of Mediconsult, had several telephone conversations with the senior management
of the third party who had expressed an interest in Mediconsult to explore the
possibility of a strategic transaction between the companies.

        An initial meeting with senior management of this third party was held
on Monday, September 18, 2000, to discuss potential terms for a transaction
between the two companies. Meetings between the two parties continued through
the balance of September and into October, 2000.

        On October 4, 2000, the Mediconsult board of directors held a special
meeting to discuss possible strategic transactions. At the meeting, Mr. Ingram
disclosed that Mediconsult had been approached by a third party about a possible
transaction between the two companies, a nondisclosure agreement had been signed
by the two companies and due diligence on the third party had begun.
"""

example3 = """
10. BACKGROUND OF THE OFFER; PAST CONTACTS OR NEGOTIATIONS WITH THE COMPANY.

In July of 2000, Foilmark retained JPMorgan, a division of Chase
Securities Inc. ("JPMorgan"), to advise on the Company's strategic alternatives.
The Company Board considered various strategic alternatives and agreed that the
Company should explore the possibility of a strategic transaction. The Company's
management and JPMorgan identified and contacted approximately 60 parties to
determine their interest in strategic transactions involving the Company. Thirty
of these parties expressed an interest in a potential transaction and were sent
confidential information memoranda after executing confidentiality agreements.
By December of 2000, eight of these parties, including Parent, submitted
preliminary indications of interest.
"""

example4 = """
BACKGROUND OF THE MERGER

On November 16, 2000, Mr. C.V. Prothro, age 58, the founder, chairman of
the board and chief executive officer of Dallas Semiconductor died suddenly. At
a meeting of Dallas Semiconductor's board of directors on November 18, 2000, the
Interim Office of the Chief Executive was formed, with its members being the
non-employee directors of Dallas Semiconductor -- Richard L. King, M.D. Sampels
and Carmelo J. Santoro. The Interim Office of the Chief Executive was created to
seek out and evaluate all available alternatives to replace the leadership lost
as a result of Mr. Prothro's death.
"""

example5 = """"
11. BACKGROUND OF THE OFFER; PAST CONTACTS, TRANSACTIONS OR NEGOTIATIONS
WITH THE COMPANY.

    On September 14, 2000, at a charity event held in Houston, Texas, Mr. John
D. Schiller, Jr., Parent's executive vice president of operations, Mr. William
L. Transier, Parent's executive vice president and chief financial officer, and
Mr. Thomas A. Reiser, a member of the Company's board of directors, had initial
discussions regarding the possibilities of Parent's acquisition of the Company.
On October 20, 2000, an initial meeting was held at Mr. Schiller's office to
further discuss such acquisition possibilities. Attendees at the meeting
included Mr. Schiller, Mr. Jerry M. Crews, an executive vice president of the
Company, and Mr. Reiser. The attendees agreed that the process of evaluation
should begin by means of a limited due diligence review of the Company by
Parent's representatives.

    On October 30, 2000, Parent and the Company executed a Confidentiality
Agreement pursuant to which the Company agreed to supply certain information to
Parent and Parent agreed to treat such information as confidential and to use
such information solely in connection with the evaluation of a possible
transaction with the Company.

    On November 6, 2000, Parent's technical team of exploration, land, legal
and business development personnel reviewed data concerning the Company's assets
with certain members of the Company's management, including Mr. Frank A.
Lodzinski, the Company's President, Mr. Crews, Mr. Tom Campbell, the Company's
manager of acquisitions, and Mr. Francis M. Mury, an executive vice president of
the Company. On November 14, 2000, an initial internal presentation was made by
Parent's technical team to Mr. Schiller regarding the information provided by
the Company.
"""

example6 = """
BACKGROUND

    For some time, the board of directors and management of Aronex
Pharmaceuticals has believed that beneficial alliances or other partnership
arrangements with significant partners would provide it with important support
and leverage in its research and development efforts, including increased
financial and personnel resources with which to develop its product portfolio.
With this in mind, in March 2000, Aronex Pharmaceuticals entered into an
agreement with Robertson Stephens, Inc. pursuant to which Robertson Stephens was
engaged to provide Aronex Pharmaceuticals with financial advisory and investment
banking services in connection with Aronex Pharmaceuticals's exploration of
various strategic alternatives, including potentially the identification and
review of possible merger candidates for, and/or acquirers of, Aronex
Pharmaceuticals. As a result, Aronex Pharmaceuticals has been in the process of
evaluating companies with complementary technologies and products under
development that would provide a good fit with its own technologies as well as
companies with substantial additional resources with which to develop its
technologies.
"""

example7 = """
BACKGROUND

    Prior to entering into merger discussions with Energy East, RGS Energy had
carefully followed the developments in the electric and natural gas industries
in the northeastern United States and, in particular, the deregulation and
restructuring of the electric and natural gas industries in New York State. In
response, RGS Energy's management and board of directors from time to time
consulted with Morgan Stanley & Co. Incorporated, RGS Energy's financial
advisor, and reviewed various strategic

alternatives, including remaining an independent public company, the possibility
of acquisitions or mergers with other companies and other transactions.

    In mid-August 2000, Wesley W. von Schack, Energy East's Chairman, President
and Chief Executive Officer, contacted Thomas S. Richards, RGS Energy's
Chairman, President and Chief Executive Officer, by telephone to discuss
informally the merits of a possible transaction between Energy East and RGS
Energy. Mr. von Schack and Mr. Richards broadly discussed the potential value
and strategic benefits that could be recognized by the shareholders of Energy
East and RGS Energy as a result of the combination of the two companies.

    On August 16, 2000, the RGS Energy board met for its annual strategy
meeting. Prior to that meeting, Mr. Richards reviewed with certain members of
the RGS Energy board his discussion with Mr. von Schack.

    Throughout September 2000, additional discussions of a preliminary nature
between Mr. von Schack and Mr. Richards periodically took place by telephone.
Such discussions generally explored the interest of each company in a possible
transaction.
"""

example8 = """
BACKGROUND OF THE MERGER

        In pursuing strategies to enhance shareholder value, both Private
Business and Towne have from time to time considered opportunities for
acquisitions, dispositions and strategic alliances. In August 1999, Bill King,
Chairman of Private Business, met in Atlanta with Drew Edwards, then the Chief
Executive Officer of Towne, to discuss the possibility of a transaction between
the companies. In early September 1999, Lynn Boggs and John Collins, directors
of Towne, and Henry Baroco, an officer and director of Towne, met in Nashville
to discuss a possible business combination with Mr. King, Tom Black, Brian
Conway, and Will Martin, directors and/or officers of Private Business. On
September 14th and 15th, 1999, a representative of Deutsche Banc Alex. Brown,
Private Business's financial advisor at the time, and Jerry Cover and Mr. Martin
from Private Business met in Atlanta with Mr. Baroco, Mr. Boggs, Glenn Sturm,
Mr. Collins, Bruce Lowthers, Cleve Shultz, and a representative of First Union
Securities, Towne's financial advisor, to conduct preliminary due diligence with
respect to the two companies. On October 13, 1999, Private Business and Towne
entered into a confidentiality agreement to allow the parties to share
additional information.
"""

example9 = """
11. BACKGROUND OF OFFER.

    On October 13, 2000, Citrix and Sequoia entered into a license agreement
pursuant to which Citrix granted Sequoia a product and trademark license to
demonstrate and market Citrix(R) NFuse(TM) portal software in connection with
Sequoia's products and services. The parties agreed to cooperate in joint
marketing and development of their respective products, and Sequoia agreed to
become a member of Citrix's business alliance program.

    In December 2000, Sequoia began considering various potential alternatives,
including possible business combinations involving Sequoia and other
corporations. In light of market conditions, Sequoia sought strategic partners
with adequate resources to assist Sequoia in strengthening its position in the
portal market.
"""

example10 = """
    (1) Background to the Offer

    In May 2000, Sara Lee announced plans to reshape its business and to look
for opportunities to acquire companies that would enhance its three major
global businesses, including its food and beverage business. In connection
with this plan, in early April 2001, Steve McMillan, President and Chief
Executive Officer of Sara Lee, telephoned Barry Beracha, Chairman and Chief
Executive Officer of Earthgrains, to request a meeting where the two might
discuss business opportunities between Sara Lee and Earthgrains.

    On April 30, Mr. McMillan met with Mr. Beracha in St. Louis to discuss
business opportunities, including a possible acquisition of Earthgrains by
Sara Lee. On May 1, Mr. McMillan met with Blair Effron of UBS Warburg LLC in
Chicago to discuss a potential transaction with Earthgrains. On May 2, Mr.
Beracha discussed his conversation with Mr. McMillan with several members of
the Board.

    On May 22, Mr. McMillan met with Mr. Beracha in New York and they agreed
that Earthgrains would provide to Sara Lee additional information and
management presentations regarding Earthgrains during the following month to
permit Sara Lee to further evaluate a possible acquisition. Mr. Beracha and
Mr. McMillan also discussed the significant terms of any transaction,
including valuation, the headquarters of Earthgrains after the consummation of
a transaction and management issues.

    On May 29, Mr. McMillan met with Mr. Effron in New York to further discuss a
possible acquisition. On the same day, Sara Lee and Earthgrains entered into a
confidentiality agreement relating to the discussions among their management
and advisors.
"""

negative_example1 = """
Background of the Merger

First Virtual Communications' Reasons for the Merger; Recommendation of the First Virtual Communications Board of Directors

CUseeMe Reasons for the Merger; Recommendation of the CUseeMe Board of Directors
"""

negative_example2 = """
The Merger  Background of the Merger;



The Merger  Packard BioSciences Reasons
for the Merger;



The Merger  Recommendation of Packard
BioSciences Board of Directors;



The Merger  PerkinElmers Reasons for the
Merger;



The Merger  Recommendation of
PerkinElmers Board of Directors;



The Merger  Opinion of PerkinElmers
Financial Advisor  Goldman, Sachs & Co.; and



The Merger  Opinion of Packard
BioSciences Financial Advisor  J.P. Morgan
Securities Inc.
"""

negative_example3 = """
and (vii) The information set forth in the sections of the Offer to Purchase entitled "Certain United States Federal Income Tax Consequences," "Background of the
Offer; Past Contacts or Negotiations with the Company," "The Transaction Documents" and "Purpose of the Offer; Plans for the Company" is incorporated herein by reference.

    (a)(2)(vi) Not
applicable.

Item 5. Past Contacts, Transactions, Negotiations and Agreements.

    The information set forth in the sections of the Offer to Purchase entitled "Certain Information Concerning Abbott and the Purchaser," "Background of the
Offer; Past Contacts or Negotiations with the Company," "The Transaction Documents" and "Purpose of the Offer; Plans for the Company" is incorporated herein by reference.

Item 6. Purpose of the Tender Offer and Plans or Proposals.

    (a),
(c)(1), (c)(3-7) The information set forth in the Introduction and in the sections of the Offer to Purchase entitled "Background of the Offer; Past Contacts or
Negotiations with the Company," "The Transaction Documents," "Purpose of the Offer; Plans for the Company," "Dividends and Distributions" and "Certain Effects of the Offer" is incorporated herein by
reference.
"""

negative_example4 = """
10.  BACKGROUND OF THE OFFER; CONTACTS WITH BEI; THE MERGER
     AGREEMENT AND RELATED AGREEMENTS............................     14

11.  PURPOSE OF THE OFFER; PLANS FOR BEI AFTER THE OFFER AND THE
     MERGER......................................................     27

12.  DIVIDENDS AND DISTRIBUTIONS.................................     29
"""

negative_example5 = """
Background of the Offer

12.
Purpose of the Offer; Plans for the Company

13.
The Merger Agreement and Other Agreements

14.
Certain Conditions of the Offer

15.
Certain Legal Matters

16.
Fees and Expenses

17.
Miscellaneous

SCHEDULE I

Directors and Executive Officers of Parent and
the Purchaser



SUMMARY TERM SHEET
"""

negative_example6 = """
OFFERSection 11Contacts and Transactions with BioReliance; Background of the
Offer, THE TENDER OFFERSection 12 Purpose of the Offer and the Merger;
Plans for BioReliance; the Merger Agreement; the Voting and Tender Agreement;
and the Confidentiality Agreement and THE TENDER OFFERSection 15Certain
Legal Matters of the Offer to Purchase is incorporated herein by reference.

     (a)(2)(vi)Not applicable.

Item 5. Past Contacts, Transactions, Negotiations and Agreements.
"""

negative_example7 = """
Contacts and Transactions with BioReliance; Background of the Offer

Purpose of the Offer and the Merger; Plans for BioReliance; the Merger Agreement; the Voting and Tender Agreement; and the Confidentiality Agreement

Dividends and Distributions

Certain Conditions of the Offer

Certain Legal Matters

Fees and Expenses

Miscellaneous

Schedule I

Directors and Executive Officers of Invitrogen and the Purchaser

I-1

Schedule II

Section 262 of the Delaware General Corporation Law

II-1

i

SUMMARY TERM SHEET

We are offering to purchase all of the outstanding common stock of
BioReliance for $48.00 net per share in cash. The following are some of the
questions you, as a stockholder of BioReliance, may have and answers to those
questions. We urge you to read carefully the remainder of this Offer to
Purchase and the accompanying Letter of Transmittal prior to making any
decision regarding your shares because the information in this summary is not
complete. Additional important information is contained in the remainder of
this Offer to Purchase and the Letter of Transmittal.

Who is offering to buy my securities?

     Our name is Baseball Acquisition Corporation. We are a Delaware
corporation formed for the purpose of making a tender offer for all of the
common stock of BioReliance and have carried on no activities other than in
connection with the merger agreement among Invitrogen, us and BioReliance. We
are a wholly-owned subsidiary of Invitrogen, a Delaware corporation listed on
the Nasdaq National Market.

     Invitrogen is a leading supplier of kits, reagents, sera and cell media
and informatics software for life sciences research, drug discovery, and the
production of biopharmaceuticals. Invitrogen offers a full range of products
that enable researchers to understand the molecular basis of life and potential
mechanisms of disease, as well as identify attractive targets for drug
development. Invitrogens products are also used to support the clinical
development and commercial production of biopharmaceuticals.
"""

negative_example8 = """
   A.  The board of directors of each of Parent, Sub and Canaan has determined
that it is in the best interests of its respective stockholders to approve the
acquisition by Parent of Canaan by means of the merger of Sub with and into
Canaan, upon the terms and subject to the conditions set forth in this
Agreement and the applicable provisions of the OGCA;

   B.  The board of directors of each of Parent, Sub and Canaan has unanimously
adopted resolutions approving the Merger, this Agreement and the transactions
contemplated hereby, and the board of directors of Canaan has unanimously
agreed to recommend that the stockholders of Canaan approve this Agreement, the
Merger and the transactions contemplated hereby;

   C.  Parent, Sub and Canaan desire to make certain representations,
warranties, covenants and agreements in connection with the Merger and also to
prescribe various conditions to the Merger;

   D.  Parent has advised Canaan and the Canaan Specified Stockholders that it
will not enter into this Agreement unless the Canaan Specified Stockholders
execute and deliver to Parent an Irrevocable Proxy in the form set forth in
Exhibit "A" attached hereto and made a part hereof;
"""


instruction = (
    "Return the start of the narrative section describing the merger or acquisition timeline. "
    "This section usually begins with 'Background of the Offer' or a similar phrase, and contains detailed events, dates, decisions, and meetings. "
    "Avoid returning boilerplate text, legal references, citations of other sections, or summaries without narrative content."
);

CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"));

def getEmbedding(text):
    response = CLIENT.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    );
    return torch.tensor(response.data[0].embedding, dtype=torch.float32);

def contrastive_query_embedding(instruction_emb, positives, negatives):
    """
        Create a query embedding from a set of positive and negative examples

        Parameters
        ----------
        instruction_emb : list[float]
            The embedding of the instruction.
        positives : list[str]
            A list of positive examples.
        negatives : list[str]
            A list of negative examples.

        Returns
        -------
        list[float]
            The query embedding.
    """
    positive_embs = torch.stack([getEmbedding(p) for p in positives]);
    negative_embs = torch.stack([getEmbedding(n) for n in negatives]);

    mean_pos = positive_embs.mean(dim=0);
    mean_neg = negative_embs.mean(dim=0);

    # Boost positives, reduce negatives
    combined = instruction_emb + mean_pos - 1.5 * mean_neg;
    return combined / combined.norm(); # normalize

def main():
    instruction_embedding = getEmbedding(instruction)
    
    positive_examples = [
        example1, example2, example3, example4, example5,
        example6, example7, example8, example9, example10
    ];

    negative_examples = [
        negative_example1, negative_example2, negative_example3,
        negative_example4, negative_example5, negative_example6,
        negative_example7, negative_example8
    ];

    queryEmbedding = contrastive_query_embedding(
        instruction_embedding,
        positive_examples,
        negative_examples
    );

    queryEmbedding_list = queryEmbedding.tolist();

    # Move this embedding json to config directory
    with open(os.path.abspath("./config/query_embedding.json"), "w") as f:
        json.dump(queryEmbedding_list, f);


if __name__ == "__main__":
    main();