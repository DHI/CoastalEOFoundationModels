"""VSCP ("very simple copy-paste"), ported from the MADOS reference `utils/vscp.py`.

Each host/donor pair yields a composite: the donor's annotated pixels (image and label)
pasted onto the host. Composites are appended to the batch.
"""

import torch


def vscp_batch(batch: dict) -> dict:
    """Append `B // 2` composites of hosts `[0:n]` and donors `[n:2n]` to the batch."""
    image, mask = batch["image"], batch["mask"]
    n = image.shape[0] // 2
    if n == 0:
        return batch

    host_image, donor_image = image[:n], image[n : 2 * n]
    host_mask, donor_mask = mask[:n], mask[n : 2 * n]
    paste = donor_mask >= 0  # negative labels are unannotated

    grown = dict(batch)
    grown["image"] = torch.cat([image, torch.where(paste.unsqueeze(1), donor_image, host_image)])
    grown["mask"] = torch.cat([mask, torch.where(paste, donor_mask, host_mask)])
    if (filenames := batch.get("filename")) is not None:
        grown["filename"] = list(filenames) + [f"{filenames[i]}+{filenames[i + n]}" for i in range(n)]
    return grown
