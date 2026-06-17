from pathlib import Path

SEROTYPES = ["DENV1", "DENV2", "DENV3", "DENV4"]

REFSEQ_FILES = {
    "DENV1": "DENV1_NC_001477.1.gb",
    "DENV2": "DENV2_NC_001474.2.gb",
    "DENV3": "DENV3_NC_001475.2.gb",
    "DENV4": "DENV4_NC_002640.1.gb",
}

FEATURE_TYPES = {"mat_peptide", "mature_protein", "peptide"}

REGION_ORDER = [
    "5'UTR",
    "C",
    "prM",
    "E",
    "NS1",
    "NS2A",
    "NS2B",
    "NS3",
    "NS4A",
    "2K",
    "NS4B",
    "NS5",
    "3'UTR",
]

PRODUCT_TO_REGION = {
    # Capsid
    "anchored capsid protein ancc": "C",
    "capsid protein c": "C",
    "capsid protein": "C",

    # prM
    "membrane glycoprotein precursor prm": "prM",
    "premembrane protein prm": "prM",
    "premembrane protein": "prM",
    "pre membrane protein": "prM",
    "membrane glycoprotein precursor": "prM",

    # E
    "envelope protein e": "E",
    "envelope protein": "E",
    "envelope glycoprotein": "E",

    # NS1
    "nonstructural protein ns1": "NS1",
    "non structural protein ns1": "NS1",
    "ns1": "NS1",

    # NS2A
    "nonstructural protein ns2a": "NS2A",
    "non structural protein ns2a": "NS2A",
    "ns2a": "NS2A",

    # NS2B
    "nonstructural protein ns2b": "NS2B",
    "non structural protein ns2b": "NS2B",
    "ns2b": "NS2B",

    # NS3
    "nonstructural protein ns3": "NS3",
    "non structural protein ns3": "NS3",
    "helicase protease ns3": "NS3",
    "ns3": "NS3",

    # NS4A
    "nonstructural protein ns4a": "NS4A",
    "non structural protein ns4a": "NS4A",
    "ns4a": "NS4A",

    # 2K
    "protein 2k": "2K",
    "2k peptide": "2K",
    "2k": "2K",

    # NS4B
    "nonstructural protein ns4b": "NS4B",
    "non structural protein ns4b": "NS4B",
    "ns4b": "NS4B",

    # NS5
    "rna dependent rna polymerase ns5": "NS5",
    "rna directed rna polymerase ns5": "NS5",
    "nonstructural protein ns5": "NS5",
    "non structural protein ns5": "NS5",
    "ns5": "NS5",
}

REGION_TO_PRODUCT = {
    "C": "anchored capsid protein ancC",
    "prM": "membrane glycoprotein precursor prM",
    "E": "envelope protein E",
    "NS1": "nonstructural protein NS1",
    "NS2A": "nonstructural protein NS2A",
    "NS2B": "nonstructural protein NS2B",
    "NS3": "nonstructural protein NS3",
    "NS4A": "nonstructural protein NS4A",
    "2K": "protein 2K",
    "NS4B": "nonstructural protein NS4B",
    "NS5": "RNA-dependent RNA polymerase NS5",
}
