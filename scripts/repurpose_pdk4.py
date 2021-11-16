from dgllife import model
from DeepPurpose.utils import name2ids, download_pretrained_model
from DeepPurpose import DTI as models
from DeepPurpose.oneliner import repurpose

if __name__ == "__main__":
    target = (
        "MKAARFVLRSAGSLNGAGLVPREVEHFSRYSPSPLSMKQLLDFGSENACERTSFAFLRQE"
        "LPVRLANILKEIDILPTQLVNTSSVQLVKSWYIQSLMDLVEFHEKSPDDQKALSDFVDTL"
        "IKVRNRHHNVVPTMAQGIIEYKDACTVDPVTNQNLQYFLDRFYMNRISTRMLMNQHILIF"
        "SDSQTGNPSHIGSIDPNCDVVAVVQDAFECSRMLCDQYYLSSPELKLTQVNGKFPDQPIH"
        "IVYVPSHLHHMLFELFKNAMRATVEHQENQPSLTPIEVIVVLGKEDLTIKISDRGGGVPL"
        "RIIDRLFSYTYSTAPTPVMDNSRNAPLAGFGYGLPISRLYAKYFQGDLNLYSLSGYGTDA"
        "IIYLKALSSESIEKLPVFNKSAFKHYQMSSEADDWCIPSREPKNLAKEVAM"
    )
    target_name = "PDK4"

    repurpose(
        target=target,
        target_name=target_name,
        pretrained_dir="./save_folder/pretrained_models",
    )
