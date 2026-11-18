package repo

import _ "embed"

// The GitHub API answers these tests feed live in testdata/, verbatim.

//go:embed testdata/commits_list.json
var fixtureCommitsList string

//go:embed testdata/issues_list.json
var fixtureIssuesList string

//go:embed testdata/issue_detail.json
var fixtureIssueDetail string

//go:embed testdata/prs_list.json
var fixturePRsList string

//go:embed testdata/pr_files.json
var fixturePRFiles string

//go:embed testdata/status_combined_failure.json
var fixtureStatusCombinedFailure string

//go:embed testdata/check_runs_mixed.json
var fixtureCheckRunsMixed string

//go:embed testdata/status_combined_failure_brief.json
var fixtureStatusCombinedFailureBrief string

//go:embed testdata/workflow_runs_one.json
var fixtureWorkflowRunsOne string

//go:embed testdata/run_jobs_two.json
var fixtureRunJobsTwo string

//go:embed testdata/workflow_runs_one_alt.json
var fixtureWorkflowRunsOneAlt string

//go:embed testdata/user_repos_two.json
var fixtureUserReposTwo string

//go:embed testdata/org_repos_two.json
var fixtureOrgReposTwo string
