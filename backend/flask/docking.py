import os
import requests
import MDAnalysis as mda
import subprocess
import prolif as plf
from rdkit import Chem
from rcsbsearchapi import rcsb_attributes as attrs
from rcsbsearchapi.search import TextQuery
import warnings
import traceback
import asyncio
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from MDAnalysis.topology import tables  # This will trigger the DeprecationWarning
from prolif.interactions import HBDonor  # This will trigger a UserWarning

warnings.filterwarnings("ignore", category=DeprecationWarning, module="MDAnalysis.topology.tables")
warnings.filterwarnings("ignore", message="The 'HBDonor' interaction has been superseded", category=UserWarning,
                        module="prolif.interactions.base")
# Add similar filters for other prolif UserWarnings if needed

# Your code that uses MDAnalysis and prolif here

def download_pdb_and_ligand(ECnumber, LIGAND_ID,
                           protein_dir="C:\\Users\\kavya\\Desktop\\MERNPS\\drugdiscovery\\backend\\flask\\protein_structures_final",
                           ligand_dir="C:\\Users\\kavya\\Desktop\\MERNPS\\drugdiscovery\\backend\\flask\\ligands_to_dock_final"):
    q1 = attrs.rcsb_polymer_entity.rcsb_ec_lineage.id == ECnumber
    q2 = TextQuery(LIGAND_ID)

    query = q1 & q2

    results = list(query())
    if not results:
        print("No results found.")
        return None, None

    pdb_id = results[0].lower()  # Get the PDB ID and convert to lowercase
    print(f"PDB ID: {pdb_id}")

    ligand_id = LIGAND_ID.lower()
    print(f"Ligand ID: {ligand_id}")

    os.makedirs(protein_dir, exist_ok=True)

    pdb_request = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb")
    ligand_request = requests.get(f"https://files.rcsb.org/ligands/download/{ligand_id}_ideal.sdf")

    print("Protein:", pdb_request.status_code)
    print("Ligand:", ligand_request.status_code)

    with open(f"{protein_dir}/{pdb_id}.pdb", "w+") as f:
        f.write(pdb_request.text)

    os.makedirs(ligand_dir, exist_ok=True)
    with open(f"{ligand_dir}/{ligand_id}_ideal.sdf", "w+") as file:
        file.write(ligand_request.text)

    return pdb_id, ligand_id


def add_hydrogens_to_protein(pdb_id):
    output_pdb_h = f"protein_structures_final/protein_{pdb_id}_h.pdb"
    input_pdb = f"protein_structures_final/protein_{pdb_id}.pdb"

    # Ensure reduce tool is installed and available in the system
    subprocess.run([
        "reduce",
        "-H",  # Adds hydrogens
        input_pdb,
        ">",  # Redirects output to a file
        output_pdb_h
    ], shell=True)
    print(f"Hydrogens added to protein {pdb_id}, saved to {output_pdb_h}")


OBABEL_PATH = r"C:\\Program Files\\OpenBabel-3.1.1\\obabel.exe"


def convert_ligand_to_pdbqt(ligand_id, sdf_file):
    try:
        output_file = f"pdbqt_final/{ligand_id}.pdbqt"
        result = subprocess.run([
            OBABEL_PATH,  # full path here
            sdf_file,
            '-O',
            output_file
        ], check=True, capture_output=True, text=True)

        print(f"[✔] obabel output:\n{result.stdout}")
        return output_file

    except subprocess.CalledProcessError as e:
        print(f"[✘] Open Babel failed:\n{e.stderr}")
        raise RuntimeError(f"Open Babel conversion failed: {e.stderr}")


import os
import traceback
import MDAnalysis as mda
import prolif as plf
from rdkit import Chem

async def run_plif_and_visualize(pdb_id, ligand_id,
                               protein_dir="C:\\Users\\kavya\\Desktop\\MERNPS\\drugdiscovery\\backend\\flask\\protein_structures_final",
                               ligand_dir="C:\\Users\\kavya\\Desktop\\MERNPS\\drugdiscovery\\backend\\flask\\ligands_to_dock_final"):
    try:
        print("🧬 Starting PLIF analysis...")

        # Step 1: Load PDB
        pdb_file = os.path.join(protein_dir, f"{pdb_id}.pdb")
        print(f"📂 Loading PDB: {pdb_file}")
        protein = mda.Universe(pdb_file)
        protein_plf = plf.Molecule.from_mda(protein, NoImplicit=False)
        print("✅ Protein loaded and converted to ProLIF format.")

        # Step 2: Prepare Ligand (add hydrogens if needed)
        input_sdf = os.path.join(ligand_dir, f"{ligand_id}_ideal.sdf")
        output_sdf = os.path.join(ligand_dir, f"{ligand_id}.sdf")

        print(f"📂 Loading ligand from: {input_sdf}")
        ligand_mol = Chem.MolFromMolFile(input_sdf, removeHs=False)
        # Chem.MolToMolFile(ligand_mol, output_sdf)
        print("✅ Ligand converted and saved with hydrogens.")

        # Step 3: Load poses
        poses_plf = list(plf.sdf_supplier(output_sdf))
        if not poses_plf:
            print("⚠️ No poses found in ligand file.")
            return None, "No poses found."

        print(f"🔢 Total ligand poses: {len(poses_plf)}")
        pose_index = 0
        ligand_pose = poses_plf[pose_index]

        # Step 4: Fingerprint
        print("🔍 Running fingerprint calculation...")
        fp = plf.Fingerprint(count=True)
        print("hi")
        fp.run_from_iterable(poses_plf, protein_plf)
        print("✅ Fingerprint calculation done.")

        if fp.ifp.empty:
            print("⚠️ PLIF returned empty fingerprint.")
            return None, "No interactions detected."

        # Step 5: Visualization
        print("🎨 Generating 3D visualization...")
        view = fp.plot_3d(ligand_pose, protein_plf, frame=0, display_all=False)
        print(f"Type of 'view' object: {type(view)}")  # Add this line
        print(f"Content of 'view' object: {view}")  # Add this line
        html_content = view._make_html()

        html_dir = os.path.join("outputs")
        os.makedirs(html_dir, exist_ok=True)
        html_path = os.path.join(html_dir, "3d_view.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"✅ 3D visualization saved at: {html_path}")
        return html_path.replace("\\", "/"), []

    except Exception as e:
        print("❌ Exception during docking workflow:", str(e))
        traceback.print_exc()
        return None, f"Error: {str(e)}"
