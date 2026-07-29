---
name: netkeiba-data-download
description: Use this skill when you need to access the netKeiba member site, download data, and save it into the workspace in a structured format.
---

# netkeiba-data-download

Use this skill for tasks involving the netKeiba member site.

## Goal
- Access the member site safely and responsibly.
- Download relevant information or files.
- Save the results into the workspace in a clear folder structure.

## Workflow
1. Confirm the target page, data type, and output folder.
2. Use a browser automation approach if login or navigation is required.
3. Download files or capture page data.
4. Save outputs under a dedicated folder such as `downloads/netkeiba/`.
5. Summarize what was downloaded and where it was saved.

## Output conventions
- Prefer a folder per date or task.
- Use descriptive filenames.
- Keep raw downloaded files and a simple summary text file together.

## Notes
- Respect the site’s terms of use and any access restrictions.
- Avoid storing secrets or credentials in the repository.
- If authentication is required, use secure local configuration and do not commit credentials.
