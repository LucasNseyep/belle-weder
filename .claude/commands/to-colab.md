# Convert Notebook to Colab

Convert a local Jupyter notebook to a Google Colab-compatible version.

## Usage
`/to-colab <path-to-notebook>`

## What to do

The user has provided a notebook path as `$ARGUMENTS`. Convert it to a Colab-compatible notebook by:

1. **Read** the source notebook at `$ARGUMENTS`

2. **Determine the output path**: prefix the filename with `colab_` (e.g. `notebooks/14_foo.ipynb` → `notebooks/colab_14_foo.ipynb`)

3. **Apply these changes** (model/training logic must stay 100% identical):

   ### At the top, add new cells:
   - **`pip-installs`**: A code cell that `!pip install`s any packages the notebook imports that aren't in the standard Colab environment (e.g. `torchmetrics`, `timm`). Check the notebook's imports to know what to install.
   - **`drive-mount`**: A code cell that follows this exact pattern (modelled on `colab_14`):
     - Declares editable variables at the top: the extracted directory path, the Drive zip path, and `expected_image_count` (total files in the dataset)
     - Defines a `count_files_with_progress(directory, description)` helper using `os.walk` + `tqdm`
     - Three-branch logic:
       1. **Dir exists and file count matches** → print ✅ skip message, do nothing
       2. **Dir exists but file count mismatches** → print ⚠️ warning, `input()` prompt asking the user whether to re-extract; if yes, mount Drive and run `!unzip -o -q "{zip_path}" -d /content`, then recount and print ✅ or ❌
       3. **Dir does not exist** → mount Drive, run `!unzip -q "{zip_path}" -d /content`, recount and print ✅ or ❌
     - Only calls `drive.mount('/content/drive')` inside the branches that actually need it (not unconditionally at the top)
     - Uses `!unzip` shell command (not Python's `zipfile`)
     - Uses ✅/⚠️/❌ status messages throughout

   ### Replace the data-indexing cells:
   - Remove any cells that load from a local CSV or local filesystem path
   - Add an **`index-images`** cell that walks the extracted `/content/my_images/` directory structure to build the image list (assumes directory = class label)

   ### Fix multiprocessing compatibility:
   - Find any `T.Lambda(lambda ...)` in transforms — replace with a named picklable class with `__init__` and `__call__`
   - Change all DataLoader `num_workers=0` to `num_workers=2`

   ### At the end, add:
   - **`save-drive`**: A cell that does `import shutil`, sets `drive_save_path = '/content/drive/MyDrive/Colab Notebooks/<checkpoint_name>.pt'`, calls `shutil.copy(CKPT, drive_save_path)`, and prints the destination. No `os.makedirs` — the Drive path must already exist.

4. **Write** the modified notebook to the output path

5. **Tell the user**:
   - The output path
   - Any `DRIVE_ZIP_PATH` or other variables they need to fill in before running
   - Any assumptions made (e.g. expected directory structure under the zip)
