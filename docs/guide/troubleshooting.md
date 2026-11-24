# Trace a failed inference request

Start at the first stage that did not produce its expected result. Save the profile ID, asset receipt, request geometry and the error from that stage.

## Device and assets {#startup}

Run `nvidia-smi` and `vflash doctor` in the actual worker environment. If the card is visible, run `vflash plan` with the same profile and GPU selection. For a rejected receipt, check changed files, paths, adapter and transformer revision. Rebuild the receipt after correcting the inventory.

## A request fails or runs out of memory {#memory}

Recheck the complete combination of mode, width, height, duration, reference count and audio recipe. Test the smallest documented request on the same profile. Record whether memory peaks during text encoding, denoising or decoding. Reduce one dimension or stage budget at a time and measure both GPU and host memory.

## The job finished but the result is wrong {#quality}

Inspect several original-size frames, including the ending, and listen to the delivered soundtrack. Check input order and first-frame versus appearance roles. For a source edit, compare the same source instants, subject scale, contact and motion; also inspect frame count, audio presence and display aspect. Report the smallest reproducible request with the runtime version and selected recipe.
