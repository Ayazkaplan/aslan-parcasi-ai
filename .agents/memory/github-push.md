---
name: GitHub push in this workspace
description: Git remotes may lack credentials even after GitHub OAuth is connected.
---

When GitHub OAuth is attached, repository writes should use the connected GitHub API proxy rather than assuming the local HTTPS git remote has credentials.

**Why:** The local remote rejected password/token authentication while the attached integration could update the repository ref successfully.

**How to apply:** Read the remote ref first, then create/update the commit through the authenticated GitHub connector without exposing credentials.