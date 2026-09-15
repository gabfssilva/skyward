// The inputs of the MCP tools this session had, from each server's tools/list
// inputSchema; written by `/plugin-types` (src/plugins/functionHooks/mcp-tool-types/mcp-tool-declarations.ts).
// Merges into the engine's ToolCallInput (types/ McpToolInputs) so
// `e.tool === "mcp__<server>__<tool>"` narrows to the tool's arguments.
// Regenerate rather than edit.
export {}
declare module 'claude-code' {
  interface McpToolInputs {
    /** Prefer `trash_message` or `mark_message_spam` instead. Adds a sensitive label (Trash or Spam) to a single message in the authenticated user's Gmail account. Use `apply_sensitive_message_label` when applying Trash or Spam to exactly 1 message. To apply sensitive labels to multiple messages, use `batch_apply_sensitive_message_labels` instead. If the message belongs to a thread that should be labeled as a whole, prefer `trash_thread` or `mark_thread_spam`. To find the message ID, use tools like `search_threads` or `get_thread`. To find the draft message ID, use tools like `list_drafts`. */
    mcp__claude_ai_Gmail__apply_sensitive_message_label: {
      /** Required. The sensitive label option to add. */
      labelOption: "LABEL_OPTION_UNSPECIFIED" | "TRASH" | "SPAM"
      /** Required. The ID of the message to add the label to. */
      messageId: string
    }
    /** Prefer `trash_thread` or `mark_thread_spam` instead. Adds a sensitive label (Trash or Spam) to a single thread in the authenticated user's Gmail account. This operation affects all messages currently in the thread. Use `apply_sensitive_thread_label` when applying Trash or Spam to exactly 1 thread. To apply sensitive labels to multiple threads, use `batch_apply_sensitive_thread_labels` instead. To find the thread ID, use the `search_threads` tool first. */
    mcp__claude_ai_Gmail__apply_sensitive_thread_label: {
      /** Required. The sensitive label option to add. */
      labelOption: "LABEL_OPTION_UNSPECIFIED" | "TRASH" | "SPAM"
      /** Required. The ID of the thread to add the label to. */
      threadId: string
    }
    /** Creates a new draft email in the authenticated user's Gmail account. This tool takes recipient addresses (`to`, `cc`, `bcc`), a `subject`, and body content as inputs. Plain text body content can be provided in `body`, and rich-text HTML content can be provided in `htmlBody` (if both are provided, `body` serves as the plain-text alternative). If the draft is created as a reply to an existing message, the ID of the original message should be passed to the tool in the `replyToMessageId` field. Returns a Draft object with the `id` and `threadId` fields populated. */
    mcp__claude_ai_Gmail__create_draft: {
      /** Optional. The attachments to include in the email. The combined size of attachments in the message cannot exceed 25MB. If you need to send files larger than 25MB, upload the file to Drive first and then insert the Drive link into `body` or `html_body`. */
      attachments?: Array<unknown /* $ref #/$defs/Attachment */>
      /** Optional. The blind carbon copy recipients of the email draft. Each string MUST be a valid plain email address (e.g., "user@example.com"). */
      bcc?: string[]
      /** Optional. The main body content of the email draft. If `html_body` is also provided, this field is treated as the plain-text alternative. */
      body?: string
      /** Optional. The carbon copy recipients of the email draft. Each string MUST be a valid plain email address (e.g., "user@example.com"). */
      cc?: string[]
      /** The HTML content of the email draft. If provided, this will be used as the rich-text version of the email. */
      htmlBody?: string
      /** Optional. The ID of the message to reply to. If provided, this will be used as the reply-to message ID for the email draft, and the `body` and `html_body` will be appended to the original message body. */
      replyToMessageId?: string
      /** Optional. The subject line of the email. Defaults to empty if not provided. */
      subject?: string
      /** Optional. The primary recipients of the email draft. Each string MUST be a valid plain email address (e.g., "user@example.com"). */
      to?: string[]
    }
    /** Creates a new label in the authenticated user's Gmail account. Supports creating nested labels (sub-labels) using a forward slash (e.g., 'Projects/Alpha/Sprint-1'). By default, parent labels will be automatically created if they do not exist. */
    mcp__claude_ai_Gmail__create_label: {
      /** Optional. Whether to automatically create parent labels for nested labels (separated by `/`). Defaults to `true`. When set to `true`, missing parent labels in the hierarchy (e.g., `Projects` and `Projects/Alpha` for `Projects/Alpha/Sprint-1`) are created automatically. When set to `false`, parent label auto-creation is disabled. */
      autoCreateParentLabels?: boolean
      /** Deprecated: Do not use. Use `color_preset` instead. Legacy field for raw text and background color hex strings. */
      color?: unknown /* $ref #/$defs/LabelColor */
      /** Optional. The color preset tile to assign to the new label. Select from predefined contrast-safe color options (e.g., LABEL_COLOR_PRESET_RED, LABEL_COLOR_PRESET_BLUE, LABEL_COLOR_PRESET_BLACK, LABEL_COLOR_PRESET_GREEN). If omitted, default label styling is applied. */
      colorPreset?: "LABEL_COLOR_PRESET_UNSPECIFIED" | "LABEL_COLOR_PRESET_BLACK" | "LABEL_COLOR_PRESET_DARK_GRAY" | "LABEL_COLOR_PRESET_GRAY" | "LABEL_COLOR_PRESET_LIGHT_GRAY" | "LABEL_COLOR_PRESET_WHITE" | "LABEL_COLOR_PRESET_RED" | "LABEL_COLOR_PRESET_ORANGE" | "LABEL_COLOR_PRESET_YELLOW" | "LABEL_COLOR_PRESET_GREEN" | "LABEL_COLOR_PRESET_MINT" | "LABEL_COLOR_PRESET_TEAL" | "LABEL_COLOR_PRESET_BLUE" | "LABEL_COLOR_PRESET_PURPLE" | "LABEL_COLOR_PRESET_PINK" | "LABEL_COLOR_PRESET_DARK_RED" | "LABEL_COLOR_PRESET_DARK_ORANGE" | "LABEL_COLOR_PRESET_DARK_GREEN" | "LABEL_COLOR_PRESET_DARK_BLUE" | "LABEL_COLOR_PRESET_DARK_PURPLE" | "LABEL_COLOR_PRESET_DARK_PINK" | "LABEL_COLOR_PRESET_BROWN"
      /** Required. The display name of the label to create. Supports nested label hierarchy using `/` (e.g., `Projects/Alpha/Sprint-1`). */
      displayName: string
      /** Optional. The visibility of the label in the label list in the Gmail web interface. Defaults to `LABEL_SHOW`. */
      labelListVisibility?: "LABEL_LIST_VISIBILITY_UNSPECIFIED" | "LABEL_SHOW" | "LABEL_SHOW_IF_UNREAD" | "LABEL_HIDE"
      /** Optional. The visibility of messages with this label in the message list in the Gmail web interface. Defaults to `SHOW`. */
      messageListVisibility?: "MESSAGE_LIST_VISIBILITY_UNSPECIFIED" | "SHOW" | "HIDE"
    }
    /** Deletes a label in the authenticated user's Gmail account. */
    mcp__claude_ai_Gmail__delete_label: {
      /** Required. The ID of the label to delete. */
      labelId: string
    }
    /** Forwards a specific email message in the authenticated user's Gmail account. Returns a Message object with the `id`, `threadId`, and `labelIds` fields populated. */
    mcp__claude_ai_Gmail__forward: {
      /** Optional. The blind carbon copy recipients of the email. Each string MUST be a valid plain email address (e.g., "user@example.com"). */
      bcc?: string[]
      /** Optional. The carbon copy recipients of the email. Each string MUST be a valid plain email address (e.g., "user@example.com"). */
      cc?: string[]
      /** Optional. Comments to add before the forwarded message. */
      forwardText?: string
      /** Optional. The HTML content of the comments to add before the forwarded message. If provided, this will be used as the rich-text version of the forward comments. */
      htmlBody?: string
      /** Required. The unique identifier of the message to forward. A specific `message_id` is required to forward, which can be obtained by retrieving the thread via `get_thread`. */
      messageId: string
      /** Optional. The primary recipients of the email. Each string MUST be a valid plain email address (e.g., "user@example.com"). */
      to?: string[]
    }
    /** Retrieves a specific draft email from the authenticated user's Gmail account by ID. The optional `messageFormat` parameter controls the format of the draft returned. Use `MINIMAL` to return snippet and key headers, `METADATA_ONLY` to exclude snippet, subject, and body, `FULL_CONTENT` for the complete draft, or `RAW` for the raw MIME message content. */
    mcp__claude_ai_Gmail__get_draft: {
      /** Required. The unique identifier of the draft to fetch. */
      draftId: string
      /** Optional. Specifies the format of the draft returned. Defaults to `FULL_CONTENT`. */
      messageFormat?: "MESSAGE_FORMAT_UNSPECIFIED" | "MINIMAL" | "FULL_CONTENT" | "METADATA_ONLY" | "PLAIN_TEXT" | "RAW"
    }
    /** Retrieves a specific email message from the authenticated user's Gmail account by its unique message ID. Use this tool to inspect a single, individual email when you already know its message ID. If the user wants to read a specific email in detail, check the exact wording of a message, or examine attachment metadata for a single email, this is the right tool. It is not suitable for retrieving entire conversations or viewing back-and-forth discussion threads; use the 'get_thread' tool instead. Note: This tool does not support retrieving draft messages. To view drafts, use the 'list_drafts' tool instead. Key indicators include if the user asks for the full content of a specific message ID returned by a previous search, or if the query asks to inspect a specific individual email rather than an entire thread. Example user prompts are: "Get the full text of message ID 18f123456789abcd.", "Read the latest message in that thread from Alice.", and "What are the attachment names in the email I just received from HR?" The optional `messageFormat` parameter controls the format of the message returned. By default (or with `FULL_CONTENT`), it returns the full content of the message. We recommend using `PLAIN_TEXT`, which returns the plain text body without the HTML body. Use `MINIMAL` to include only subject and snippet (excluding body). Use `METADATA_ONLY` to include only basic metadata (message ID, thread ID, labels, timestamp, and size estimate). */
    mcp__claude_ai_Gmail__get_message: {
      /** Optional. Specifies the format of the message returned. Defaults to `FULL_CONTENT`. We recommend using `PLAIN_TEXT` to prevent context exhaustion. */
      messageFormat?: "MESSAGE_FORMAT_UNSPECIFIED" | "MINIMAL" | "FULL_CONTENT" | "METADATA_ONLY" | "PLAIN_TEXT" | "RAW"
      /** Required. The unique identifier of the message to fetch. */
      messageId: string
    }
    /** Retrieves a specific email thread from the authenticated user's Gmail account, including a list of its messages. Note: This tool does not support retrieving drafts. Any draft messages within a thread are omitted. To view drafts, use the `list_drafts` tool instead. The optional `messageFormat` parameter controls the format of the messages returned. By default (or with `FULL_CONTENT`), it returns the full content of messages. We recommend using `PLAIN_TEXT`, which returns the plain text body without the HTML body. Use `MINIMAL` to include only subject and snippet (excluding body). Use `METADATA_ONLY` to include only basic metadata (message ID, thread ID, labels, timestamp, and size estimate). */
    mcp__claude_ai_Gmail__get_thread: {
      /** Optional. Specifies the format of the messages returned within the thread. Defaults to `FULL_CONTENT`. We recommend using `PLAIN_TEXT` to prevent context exhaustion. Note: `MINIMAL` format returns `id`, `snippet`, `subject`, `sender`, `to_recipients`, `cc_recipients`, `bcc_recipients`, `date`, `label_ids`. `METADATA_ONLY` format returns `id`, `sender`, `to_recipients`, `cc_recipients`, `bcc_recipients`, `date`, `label_ids`. `FULL_CONTENT` returns `id`, `snippet`, `subject`, `sender`, `to_recipients`, `cc_recipients`, `bcc_recipients`, `date`, `label_ids`, `attachment_ids`, `plaintext_body`, `html_body`, `attachments`. `PLAIN_TEXT` returns `id`, `snippet`, `subject`, `sender`, `to_recipients`, `cc_recipients`, `bcc_recipients`, `date`, `label_ids`, `attachment_ids`, `plaintext_body`, `attachments` (without `html_body`). `RAW` format is not supported here. */
      messageFormat?: "MESSAGE_FORMAT_UNSPECIFIED" | "MINIMAL" | "FULL_CONTENT" | "METADATA_ONLY" | "PLAIN_TEXT" | "RAW"
      /** Required. The unique identifier of the thread to fetch. */
      threadId: string
    }
    /** Adds one or more labels to a specific message in the authenticated user's Gmail account. To find the message ID, use tools like `search_threads` or `get_thread`. If unsure of a user label's ID, use the `list_labels` tool first to discover available labels and their IDs. To move a specific message to Trash or mark it as Spam, please use the `trash_message` or `mark_message_spam` tool instead. */
    mcp__claude_ai_Gmail__label_message: {
      /** Required. The IDs of the labels to add. Can be a system label ID (e.g., `INBOX`, `STARRED`, `UNREAD`, `IMPORTANT`) or a user-defined label ID. The tool accepts `label_ids` and not label names. Use the `list_labels` tool to get the corresponding label id to a display name for user-defined labels. */
      labelIds: string[]
      /** Required. The ID of the message to add the labels to. */
      messageId: string
    }
    /** Adds labels to an entire thread in the authenticated user's Gmail account. This operation affects all messages currently in the thread and any future messages added to it. If unsure of the thread ID, use the `search_threads` tool first. If unsure of a user label's ID, use the `list_labels` tool first to discover available labels and their IDs. To move a thread to Trash or mark it as Spam, please use the `trash_thread` or `mark_thread_spam` tool instead. */
    mcp__claude_ai_Gmail__label_thread: {
      /** Required. The unique identifiers of the labels to add. Can be a system label ID (e.g., `INBOX`, `STARRED`, `UNREAD`, `IMPORTANT`) or a user-defined label ID. The tool accepts `label_ids` and not label names. Use the `list_labels` tool to get the corresponding label id to a display name for user-defined labels. */
      labelIds: string[]
      /** Required. The unique identifier of the thread to add labels to. */
      threadId: string
    }
    /** Lists draft emails from the authenticated user's Gmail account. This tool can filter drafts based on a query string and supports pagination. It returns a list of drafts, including their IDs and subjects (unless `view` is set to `DRAFT_VIEW_METADATA_ONLY`). `page_token` can be used to paginate the results. To retrieve subsequent pages of results, use the `page_token` returned in the previous response. The `view` parameter controls which fields are populated in the response. By default (or with `DRAFT_VIEW_FULL`), it returns full content. Use `DRAFT_VIEW_METADATA_ONLY` to exclude sensitive content like subject and body. Note: An empty JSON object `{}` represents zero matching items, not an error. */
    mcp__claude_ai_Gmail__list_drafts: {
      /** Optional. The maximum number of drafts to return. If unspecified, defaults to 20. The maximum allowed value is 50. */
      pageSize?: number
      /** Optional. A token received from a previous `list_drafts` call to retrieve the next page of results. Leave empty to fetch the first page. This is primarily used for pagination to continue fetching results from where the previous `ListDraft` call left off, especially when the number of drafts matching the query exceeds the `page_size` limit. */
      pageToken?: string
      /** Examples: - `subject:OneMCP Update` - `from:gduser1@workspacesamples.dev` - `to:gduser2@workspacesamples.dev AND newer_than:7d` - `project proposal has:attachment` - `is:unread` A space or a dash (`-`) will separate a number while a dot (`.`) will be a decimal. For example, `01.2047-100` is considered two numbers: `01.2047` and `100`. Note: If we want to ensure all drafts for the query are returned, we can paginate the results by making repeated calls to the tool until the response contains an empty list of drafts. */
      query?: string
      /** Optional. Controls the fields populated for drafts in the draft list. Defaults to returning metadata only (`id`, `thread_id`, `to_recipients`, `cc_recipients`, `bcc_recipients`, `date`). Set to `DRAFT_VIEW_FULL` to include `subject` and `plaintext_body` content. */
      view?: "DRAFT_VIEW_UNSPECIFIED" | "DRAFT_VIEW_METADATA_ONLY" | "DRAFT_VIEW_FULL"
    }
    /** Lists all labels available in the authenticated user's Gmail account. Use this tool to discover the `id` of a label before calling `label_thread`, `unlabel_thread`, `label_message`, or `unlabel_message`. Note: the system labels, `DRAFT` and `SENT`, cannot be set on messages and are read only. Note: An empty JSON object `{}` represents zero matching items, not an error. */
    mcp__claude_ai_Gmail__list_labels: {}
    /** Marks a specific message as Spam in the authenticated user's Gmail account. To find the message ID, use tools like `search_threads` or `get_thread`. */
    mcp__claude_ai_Gmail__mark_message_spam: {
      /** Required. The ID of the message to mark as Spam. */
      messageId: string
    }
    /** Marks an entire thread as Spam in the authenticated user's Gmail account. This operation affects all messages currently in the thread. Use `mark_thread_spam` when marking a thread as spam, even if it currently contains only 1 message. Marking spam at the thread level ensures all current messages in the thread are marked as Spam. If unsure of the thread ID, use the `search_threads` tool first. */
    mcp__claude_ai_Gmail__mark_thread_spam: {
      /** Required. The ID of the thread to mark as Spam. */
      threadId: string
    }
    /** Replies to a specific email message in the authenticated user's Gmail account. Supports replying to only the sender or to all recipients (reply-all) via the `replyAll` parameter. Requires the `messageId` of the message to reply to. If `htmlBody` is not provided, then `body` is required. If `body` is not provided, then `htmlBody` is required. To reply to an existing thread, retrieve the thread via `get_thread` first to find the `messageId` of the latest message in that thread. Returns a Message object with the `id`, `threadId`, and `labelIds` fields populated. */
    mcp__claude_ai_Gmail__reply: {
      /** Optional. The blind carbon copy recipients of the email reply. Each string MUST be a valid plain email address (e.g., "user@example.com"). */
      bcc?: string[]
      /** Optional. The main body content of the reply in plain text. If `html_body` is also provided, this field is treated as the plain-text alternative. If `html_body` is not provided, then `body` is required. */
      body?: string
      /** Optional. The carbon copy recipients of the email reply. If specified, overrides the default CC recipients. Each string MUST be a valid plain email address (e.g., "user@example.com"). */
      cc?: string[]
      /** Optional. The HTML content of the reply. If provided, this will be used as the rich-text version of the email. If `body` is not provided, then `html_body` is required. */
      htmlBody?: string
      /** Required. The unique identifier of the message to reply to. If you want to reply to an existing thread, first retrieve the thread via `get_thread` to find the `message_id` of the last message in the thread. Pass that `message_id` here to ensure proper threading. */
      messageId: string
      /** Optional. Whether to reply to all recipients. Defaults to false. */
      replyAll?: boolean
      /** Optional. The primary recipients of the email reply. If specified, overrides the default reply recipients. Each string MUST be a valid plain email address (e.g., "user@example.com"). */
      to?: string[]
    }
    /** Lists email threads from the authenticated user's Gmail account. This tool can filter threads based on a query string and supports pagination. It returns a list of threads, including their IDs and related messages. Each related message contains details like a snippet of the message body, the subject, the sender, the recipients etc. The `view` parameter controls which fields are populated in the related messages. By default (or with `THREAD_VIEW_MINIMAL`), it includes subject and snippet. Use `THREAD_VIEW_METADATA_ONLY` to exclude subject and snippet. Note that the full message bodies are not returned by this tool; use the 'get_thread' tool with a thread ID to fetch the full message body if needed. Threads with excluded criteria may still appear in the results. This occurs because Gmail identifies matching messages first. For example, if you search for -is:starred, Gmail will find an entire thread if it contains at least one unstarred message, even if other emails in that same conversation are starred. Note: An empty JSON object `{}` represents zero matching items, not an error. */
    mcp__claude_ai_Gmail__search_threads: {
      /** Optional. Include threads from TRASH in the results. Defaults to false. */
      includeTrash?: boolean
      /** Optional. The maximum number of threads to return. If unspecified, defaults to 20. The maximum allowed value is 50. */
      pageSize?: number
      /** Optional. Page token to retrieve a specific page of results in the list. Leave empty to fetch the first page. This is primarily used for pagination to continue fetching results from where the previous `SearchThreads` call left off, especially when the number of threads matching the query exceeds the `page_size` limit. */
      pageToken?: string
      /** Optional. A query string to filter the threads. Natural language queries must be pre-converted into Gmail syntax queries to use this tool. If omitted, all threads (excluding spam and trash by default) are listed. Supported Operators by Category: Sender & Recipient: - `from:` — Sent from a specific person. - `to:` — Sent to a specific person. - `cc:` — Specific people in Cc. - `bcc:` — Specific people in Bcc. - `deliveredto:` — Delivered to a specific address. - `list:` — From a specific mailing list. Time & Date: - `after:YYYY/MM/DD` / `newer:YYYY/MM/DD` — Received after a date. - `before:YYYY/MM/DD` / `older:YYYY/MM/DD` — Received before a date. - `older_than:` — Older than a duration (for example, `1y`, `2d`). - `newer_than:` — Newer than a duration. Content: - `subject:` — Words in the subject line. - `has:` — Has specific content types (attachment, drive, youtube, document). - `filename:` — Attachment with a specific name or type. - `""` — Search for an exact word or phrase. (for example, `"holiday"`, `"holiday vacation"`). Note: Double quotes enforce strict contiguous phrase matching. For topic, discussion, or keyword queries, prefer unquoted keywords (e.g. `partner advertising` instead of `"partner advertising"`). - `+` — Match a word exactly. (for example, `+holiday`, `+unicorn`) - `rfc822msgid:` — Specific message ID header. - `AROUND ` — Find words near each other (for example, `holiday AROUND 10 vacation`). Labels & Categories: - `label:` — Under a specific label. The tool accepts label IDs, not display names. Use the `list_labels` tool to get the ID. - `category:` — In a category (primary, social, promotions, updates, forums, reservations, purchases). - `in:` — Search in specific labels (archive, snoozed, trash, sent, inbox). For example, `in:trash`, `in:inbox`. Archived and sent messages are included by default; use `-in:archive` and `-in:sent` to exclude them. Drafts are explicitly excluded by default by the tool. Use `in:inbox` to restrict search to the inbox only. - `has:userlabels` — Has any user labels. - `has:nouserlabels` — Does not have any user labels. - `has:*-star` — Specific star colors (if enabled, for example, `has:yellow-star`). - `in:draft` — Search in drafts. -in:draft means exclude drafts from the search results. - `in:sent` — Search in sent messages. - `in:anywhere` — Search in all folders (including spam and trash). Status: - `is:` — Search by status (important, starred, unread, read, muted). Size: - `size:` — Specific size in bytes. - `larger:` / `smaller:` — Larger or smaller than a size (for example, `10M` for 10 MB). Logic & Grouping: - `AND` — Match all criteria (default behavior). - `OR` or `{ }` — Match one or more criteria (for example, `from:amy OR from:david`, `{from:amy from:david}`). - `-` (minus) — Exclude criteria (for example, `-movie`). - `( )` — Group multiple search terms (for example, `subject:(dinner film)`). Examples: - `subject:OneMCP Update` - `from:user@example.com` - `to:user2@example.com AND newer_than:7d` - `project proposal has:attachment` - `is:unread -in:draft` To prevent overly strict queries, favor concise, keyword-based queries over long subject strings or full sentences. Avoid copying overly detailed subjects from the user prompt verbatim, as this often leads to search misses. Instead, extract the most unique keywords (e.g., subject:amazon \"delivery\" OR \"order\" instead of \"amazon order\"). Use boolean operators to broaden your search coverage. Use OR to search for synonyms or multiple potential senders, and use ( ) for grouping criteria. Note that whitespace between terms acts as an implicit AND. */
      query?: string
      /** Optional. Controls the fields populated for threads in the thread list. Defaults to `THREAD_VIEW_MINIMAL`. `THREAD_VIEW_MINIMAL` returns `id`, `snippet`, `subject`, `sender`, `to_recipients`, `cc_recipients`, `bcc_recipients`, `date`, `label_ids`. `THREAD_VIEW_METADATA_ONLY` returns `id`, `sender`, `to_recipients`, `cc_recipients`, `bcc_recipients`, `date`, `label_ids`. */
      view?: "THREAD_VIEW_UNSPECIFIED" | "THREAD_VIEW_METADATA_ONLY" | "THREAD_VIEW_MINIMAL"
    }
    /** Sends a new email message immediately from the authenticated user's Gmail account. To send an existing draft message, provide the `draftId`. To send a new message, provide recipients in `to`, `cc`, or `bcc`, a `subject`, and message content in `body` or `htmlBody`. To thread the message under an existing thread or conversation, provide `replyThreadId` (preferred for send-only clients) or `replyToMessageId`. If sending a new message, attachments can be included via the `attachments` field, but the combined size cannot exceed 25MB. The email can be a previously created draft (identified by `draftId`) or a new email with provided recipients `to`, `cc`, and `bcc`, `subject` and `body` content (including plain text and HTML). Returns a Message object with the `id`, `threadId`, and `labelIds` fields populated. */
    mcp__claude_ai_Gmail__send_message: {
      /** Optional. The attachments to include in the email. The combined size of attachments in the message cannot exceed 25MB. If you need to send files larger than 25MB, upload the file to Drive first and then insert the Drive link into `body` or `html_body`. */
      attachments?: Array<unknown /* $ref #/$defs/Attachment */>
      /** Optional. The blind carbon copy recipients of the email. Each string MUST be a valid plain email address (e.g., "user@example.com"). */
      bcc?: string[]
      /** Optional. The main body content of the email. If `html_body` is also provided, this field is treated as the plain-text alternative. */
      body?: string
      /** Optional. The carbon copy recipients of the email. Each string MUST be a valid plain email address (e.g., "user@example.com"). */
      cc?: string[]
      /** Optional. The unique identifier of an existing draft to send. If provided, the other fields (`to`, `cc`, `bcc`, `subject`, `body`, `html_body`) are ignored, and the specified draft is sent as is. */
      draftId?: string
      /** Optional. The HTML content of the email. If provided, this will be used as the rich-text version of the email. */
      htmlBody?: string
      /** Optional. The unique identifier of the thread to send this message in. If provided, the sent message will be threaded under the specified thread. Compatible with all scopes including send-only (gmail.send). */
      replyThreadId?: string
      /** Optional. The unique identifier of the message to reply to. If provided, this message will be threaded in reply to the specified message. Note: Resolving a message by ID requires read permissions (e.g., 'gmail.modify' or 'gmail.compose'). If the caller only has send-only permissions ('gmail.send'), use `reply_thread_id` instead. */
      replyToMessageId?: string
      /** Optional. The subject line of the email. */
      subject?: string
      /** Optional. The primary recipients of the email. Required if `draft_id` is not provided. Each string MUST be a valid plain email address (e.g., "user@example.com"). */
      to?: string[]
    }
    /** Moves a specific message to the Trash in the authenticated user's Gmail account. Use `trash_message` when targeting a specific message within a thread. To trash an entire thread or a single-message thread, prefer `trash_thread`. To find the message ID, use tools like `search_threads` or `get_thread`. To find the draft message ID, use tools like `list_drafts`. */
    mcp__claude_ai_Gmail__trash_message: {
      /** Required. The ID of the message to move to Trash. */
      messageId: string
    }
    /** Moves an entire thread to the Trash in the authenticated user's Gmail account. This operation affects all messages currently in the thread. Use `trash_thread` when trashing a thread, even if it currently contains only 1 message. Trashing at the thread level ensures all current messages in the thread are moved to Trash. If unsure of the thread ID, use the `search_threads` tool first. */
    mcp__claude_ai_Gmail__trash_thread: {
      /** Required. The ID of the thread to move to Trash. */
      threadId: string
    }
    /** Removes one or more labels from a specific message in the authenticated user's Gmail account. To find the message ID, use tools like `search_threads` or `get_thread`. If unsure of a user label's ID, use the `list_labels` tool first to discover available labels and their IDs. */
    mcp__claude_ai_Gmail__unlabel_message: {
      /** Required. The IDs of the labels to remove. Can be a system label ID (e.g., `INBOX`, `TRASH`, `SPAM`, `STARRED`, `UNREAD`, `IMPORTANT`) or a user-defined label ID. The tool accepts `label_ids` and not label names. Use the `list_labels` tool to get the corresponding label id to a display name for user-defined labels. */
      labelIds: string[]
      /** Required. The ID of the message to remove the labels from. */
      messageId: string
    }
    /** Removes labels from an entire thread in the authenticated user's Gmail account. If unsure of the thread ID, use the `search_threads` tool first. If unsure of a user label's ID, use the `list_labels` tool first. */
    mcp__claude_ai_Gmail__unlabel_thread: {
      /** Required. The unique identifiers of the labels to remove. Can be a system label ID (e.g., `INBOX`, `TRASH`, `SPAM`, `STARRED`, `UNREAD`, `IMPORTANT`) or a user-defined label ID. The tool accepts `label_ids` and not label names. Use the `list_labels` tool to get the corresponding label id to a display name for user-defined labels. */
      labelIds: string[]
      /** Required. The unique identifier of the thread to remove labels from. */
      threadId: string
    }
    /** Unmarks a specific message as Spam in the authenticated user's Gmail account. To find the message ID, use tools like `search_threads` or `get_thread`. */
    mcp__claude_ai_Gmail__unmark_message_spam: {
      /** Required. The ID of the message to unmark as Spam. */
      messageId: string
    }
    /** Unmarks an entire thread as Spam in the authenticated user's Gmail account. If unsure of the thread ID, use the `search_threads` tool first. */
    mcp__claude_ai_Gmail__unmark_thread_spam: {
      /** Required. The ID of the thread to unmark as Spam. */
      threadId: string
    }
    /** Removes a specific message from the Trash in the authenticated user's Gmail account. To find the message ID, use tools like `search_threads` or `get_thread`. */
    mcp__claude_ai_Gmail__untrash_message: {
      /** Required. The ID of the message to remove from Trash. */
      messageId: string
    }
    /** Removes an entire thread from the Trash in the authenticated user's Gmail account. If unsure of the thread ID, use the `search_threads` tool first. */
    mcp__claude_ai_Gmail__untrash_thread: {
      /** Required. The ID of the thread to remove from Trash. */
      threadId: string
    }
    /** Updates an existing draft email in the authenticated user's Gmail account. This operation supports merge semantics: fields provided in the request (non-empty) will overwrite the corresponding fields in the draft, while omitted (or empty) fields will preserve their existing values. WARNING: Attachments are NOT merged. If the draft contains attachments, they will be removed unless they are explicitly re-provided in the `attachments` field of this request. Returns a Draft object with the `id` and `threadId` fields populated. */
    mcp__claude_ai_Gmail__update_draft: {
      /** Optional. The attachments to include in the email. The combined size of attachments in the message cannot exceed 25MB. If you need to send files larger than 25MB, upload the file to Drive first and then insert the Drive link into `body` or `html_body`. If omitted or empty, any existing attachments on the draft will be removed. */
      attachments?: Array<unknown /* $ref #/$defs/Attachment */>
      /** Optional. The blind carbon copy recipients of the email draft. Each string MUST be a valid plain email address (e.g., "user@example.com"). If omitted or empty, the existing recipients are preserved. */
      bcc?: string[]
      /** Optional. The main body content of the email draft. If `html_body` is also provided, this field is treated as the plain-text alternative. If both `body` and `html_body` are omitted or empty, the existing body is preserved. If `body` is provided but `html_body` is omitted, the body will be updated to plain text and the existing HTML body will be cleared. */
      body?: string
      /** Optional. The carbon copy recipients of the email draft. Each string MUST be a valid plain email address (e.g., "user@example.com"). If omitted or empty, the existing recipients are preserved. */
      cc?: string[]
      /** Required. The unique identifier of the draft to update. */
      draftId: string
      /** Optional. The HTML content of the email draft. If provided, this will be used as the rich-text version of the email. If both `body` and `html_body` are omitted or empty, the existing body is preserved. If `html_body` is provided but `body` is omitted, the body will be updated to HTML and the existing plain text body will be cleared. */
      htmlBody?: string
      /** Optional. The subject line of the email. If omitted or empty, the existing subject is preserved. */
      subject?: string
      /** Optional. The primary recipients of the email draft. Each string MUST be a valid plain email address (e.g., "user@example.com"). If omitted or empty, the existing recipients are preserved. */
      to?: string[]
    }
    /** Modifies an existing label's name and color in the user's Gmail account. */
    mcp__claude_ai_Gmail__update_label: {
      /** Deprecated: Do not use. Use `color_preset` instead. Legacy field for raw text and background color hex strings. */
      color?: unknown /* $ref #/$defs/LabelColor */
      /** Optional. The new color preset tile to assign to the label. Select from predefined contrast-safe color options (e.g., LABEL_COLOR_PRESET_RED, LABEL_COLOR_PRESET_BLUE, LABEL_COLOR_PRESET_BLACK, LABEL_COLOR_PRESET_GREEN). If omitted, existing label color is preserved. */
      colorPreset?: "LABEL_COLOR_PRESET_UNSPECIFIED" | "LABEL_COLOR_PRESET_BLACK" | "LABEL_COLOR_PRESET_DARK_GRAY" | "LABEL_COLOR_PRESET_GRAY" | "LABEL_COLOR_PRESET_LIGHT_GRAY" | "LABEL_COLOR_PRESET_WHITE" | "LABEL_COLOR_PRESET_RED" | "LABEL_COLOR_PRESET_ORANGE" | "LABEL_COLOR_PRESET_YELLOW" | "LABEL_COLOR_PRESET_GREEN" | "LABEL_COLOR_PRESET_MINT" | "LABEL_COLOR_PRESET_TEAL" | "LABEL_COLOR_PRESET_BLUE" | "LABEL_COLOR_PRESET_PURPLE" | "LABEL_COLOR_PRESET_PINK" | "LABEL_COLOR_PRESET_DARK_RED" | "LABEL_COLOR_PRESET_DARK_ORANGE" | "LABEL_COLOR_PRESET_DARK_GREEN" | "LABEL_COLOR_PRESET_DARK_BLUE" | "LABEL_COLOR_PRESET_DARK_PURPLE" | "LABEL_COLOR_PRESET_DARK_PINK" | "LABEL_COLOR_PRESET_BROWN"
      /** Optional. The human-readable display name of the label. */
      displayName?: string
      /** Required. The unique identifier of the label to modify. Use the `list_labels` tool to get the corresponding label id to a display name for user-defined labels. */
      labelId: string
      /** Optional. The new visibility of the label in the label list in the Gmail web interface. */
      labelListVisibility?: "LABEL_LIST_VISIBILITY_UNSPECIFIED" | "LABEL_SHOW" | "LABEL_SHOW_IF_UNREAD" | "LABEL_HIDE"
      /** Optional. The new visibility of messages with this label in the message list in the Gmail web interface. */
      messageListVisibility?: "MESSAGE_LIST_VISIBILITY_UNSPECIFIED" | "SHOW" | "HIDE"
    }
    /** Atomically adds and/or removes labels from a specific message in the authenticated user's Gmail account. Requires at least one of `addLabelIds` or `removeLabelIds` to be provided. Moving an email between labels can be accomplished in a single call by specifying the target label in `addLabelIds` and the current label in `removeLabelIds`. */
    mcp__claude_ai_Gmail__update_message_labels: {
      /** Optional. The IDs of the labels to add. Can be a system label ID (e.g., `INBOX`, `STARRED`, `UNREAD`, `IMPORTANT`) or a user-defined label ID. */
      addLabelIds?: string[]
      /** Required. The ID of the message to modify labels for. */
      messageId: string
      /** Optional. The IDs of the labels to remove. Can be a system label ID or a user-defined label ID. */
      removeLabelIds?: string[]
    }
    /** Creates an event on the given calendar. */
    mcp__claude_ai_Google_Calendar__create_event: {
      /** Optional. Create and add a Google Meet URL. Default: `false`. */
      addGoogleMeetUrl?: boolean
      /** Optional. Whether the event spans the entire day. If true, start/end times are treated as midnight. */
      allDay?: boolean
      /** Optional. File attachments. */
      attachments?: Array<unknown /* $ref #/$defs/Attachment */>
      /** Optional. Deprecated: use `attendees` instead. */
      attendeeEmails?: string[]
      /** Optional. Attendees of the event. For events that are created on the user's primary calendar with at least one other attendee, the current user will automatically be added as an attendee if not already included. */
      attendees?: Array<unknown /* $ref #/$defs/Attendee */>
      /** Optional. Availability setting. */
      availability?: "AVAILABILITY_UNSPECIFIED" | "AVAILABILITY_BUSY" | "AVAILABILITY_FREE"
      /** Optional. ID of the calendar to create the event on. Email address - can be resolved using `list_calendars`. Default: primary calendar. */
      calendarId?: string
      /** Optional. The color of the event. For a list of color IDs, refer to the documentation of the Event resource. */
      colorId?: string
      /** Optional. Description. Can contain HTML. */
      description?: string
      /** Required. End time (ISO 8601, for example `2026-04-30T11:00:00+08:00`). */
      endTime: string
      /** Optional. Type of the event. */
      eventType?: "EVENT_TYPE_UNSPECIFIED" | "DEFAULT" | "OUT_OF_OFFICE" | "FOCUS_TIME" | "WORKING_LOCATION" | "BIRTHDAY" | "FROM_GMAIL"
      /** Optional. Specific Google Meet URL or meeting ID. Overrides `add_google_meet_url`. */
      googleMeetUrl?: string
      /** Optional. Guest permissions. */
      guestPermissions?: unknown /* $ref #/$defs/GuestPermissions */
      /** Optional. Location. */
      location?: string
      /** Optional. Which email notification should be sent for this event update. */
      notificationLevel?: "NOTIFICATION_LEVEL_UNSPECIFIED" | "NONE" | "EXTERNAL_ONLY" | "ALL"
      /** Optional. Reminders override calendar defaults. */
      overrideReminders?: Array<unknown /* $ref #/$defs/Reminder */>
      /** Optional. Recurrence rules as `RRULE`, `RDATE`, or `EXDATE` strings (per RFC 5545). */
      recurrenceData?: string[]
      /** Required. Start time (ISO 8601, for example `2026-04-30T10:00:00+08:00`). */
      startTime: string
      /** Required. Title. */
      summary: string
      /** Optional. IANA Time Zone Database name (for example, `America/Los_Angeles`). Default: the user's primary time zone. Overrides offsets in `start_time` and `end_time`. */
      timeZone?: string
      /** Optional. Visibility of the event. Possible values are: - `default` - Uses the default visibility for events on the calendar. Default value. - `public` - The event is public and event details are visible to all readers of the calendar. - `private` - Only event attendees may view event details. */
      visibility?: string
      /** Optional. Working location properties (if `eventType` is `WORKING_LOCATION`). */
      workingLocationProperties?: unknown /* $ref #/$defs/WorkingLocationProperties */
    }
    /** Deletes an event on the given calendar. */
    mcp__claude_ai_Google_Calendar__delete_event: {
      /** Optional. ID of the calendar containing the event. Email address - can be resolved using `list_calendars`. Default: primary calendar. */
      calendarId?: string
      /** Required. The ID of the event to delete. */
      eventId: string
      /** Optional. Which email notification should be sent for this event update. */
      notificationLevel?: "NOTIFICATION_LEVEL_UNSPECIFIED" | "NONE" | "EXTERNAL_ONLY" | "ALL"
    }
    /** Returns a single event on the given calendar. */
    mcp__claude_ai_Google_Calendar__get_event: {
      /** Optional. ID of the calendar containing the event. Email address - can be resolved using `list_calendars`. Default: primary calendar. */
      calendarId?: string
      /** Required. Event ID. Can be resolved using `list_events` or `search_events`. */
      eventId: string
    }
    /** Returns the calendars this user has access to (their calendar list). Use this tool to resolve calendar identifying data (for example, 'my family calendar') into its corresponding `calendar_id` (email identifier) */
    mcp__claude_ai_Google_Calendar__list_calendars: {
      /** Optional. Max results per page. Default `100`, max `250`. */
      pageSize?: number
      /** Optional. Token specifying which result page to return. */
      pageToken?: string
    }
    /** Returns events on the given calendar matching all specified constraints. Time constraints should not be specified unless requested by the user. For open-ended keyword or topic-based searches on the primary calendar, the search_events tool must be used instead. */
    mcp__claude_ai_Google_Calendar__list_events: {
      /** Optional. ID of the calendar containing the events. Email address - can be resolved using `list_calendars`. Default: primary calendar. */
      calendarId?: string
      /** Optional. The upper bound of a time range. Must only be set when a specific timeframe or a time in the past is requested by the user. Must be an ISO 8601 timestamp greater than `start_time`. */
      endTime?: string
      /** Optional. The event types to return. If empty, only the following event types are returned: `DEFAULT`, `OUT_OF_OFFICE`, `FOCUS_TIME`, `FROM_GMAIL` */
      eventType?: Array<"EVENT_TYPE_UNSPECIFIED" | "DEFAULT" | "OUT_OF_OFFICE" | "FOCUS_TIME" | "WORKING_LOCATION" | "BIRTHDAY" | "FROM_GMAIL">
      /** Optional. Deprecated: use `event_type` instead. */
      eventTypeFilter?: string[]
      /** Optional. Free-form case-insensitive search matching title, description, location, or attendees. Matches events containing all query terms verbatim (AND search). */
      fullText?: string
      /** Optional. The order in which events should be returned. Possible values are: - `default` - Unspecified, but deterministic ordering (default). - `startTime` - Order by start time ascending. - `startTimeDesc` - Order by start time descending. - `lastModified` - Order by last modification time ascending. */
      orderBy?: string
      /** Optional. Max events per page (default `100`, max `250`). Recommended: `10`. */
      pageSize?: number
      /** Optional. Next page token. Use the value from the previous page's `nextPageToken`. */
      pageToken?: string
      /** Optional. The lower bound of a time range. Must only be set when a specific timeframe is requested by the user. Must be an ISO 8601 timestamp less than `end_time`. */
      startTime?: string
      /** Optional. Time zone (IANA ID, for example `Europe/Zurich`) used to resolve timezone-less dates. Default: calendar's timezone. */
      timeZone?: string
    }
    /** Responds to an event on a calendar. */
    mcp__claude_ai_Google_Calendar__respond_to_event: {
      /** Optional. ID of the calendar containing the event. Email address - can be resolved using `list_calendars`. Default: primary calendar. */
      calendarId?: string
      /** Required. The ID of the event to respond to. */
      eventId: string
      /** Optional. Which email notification should be sent for this event update. */
      notificationLevel?: "NOTIFICATION_LEVEL_UNSPECIFIED" | "NONE" | "EXTERNAL_ONLY" | "ALL"
      /** Optional. The user's comment attached to the response. */
      responseComment?: string
      /** Required. The new user's response status of the event. Possible values are: - `declined` - The attendee has declined the invitation. - `tentative` - The attendee has tentatively accepted the invitation. - `accepted` - The attendee has accepted the invitation. */
      responseStatus: string
    }
    /** Searches events on the user's primary calendar using semantic search. */
    mcp__claude_ai_Google_Calendar__search_events: {
      /** Optional. Maximum number of entries returned on one result page. */
      pageSize?: number
      /** Optional. Token specifying which result page to return. */
      pageToken?: string
      /** Required. Query string to search for events (case-insensitive). */
      query: string
    }
    /** Suggests time periods across one or more calendars. */
    mcp__claude_ai_Google_Calendar__suggest_time: {
      /** Required. Attendee emails to find free time for. */
      attendeeEmails: string[]
      /** Optional. Min duration of free slot in minutes. Default: `30`. */
      durationMinutes?: number
      /** Required. Query interval end (ISO 8601). */
      endTime: string
      /** Preferences to find suggested time. */
      preferences?: unknown /* $ref #/$defs/Preferences */
      /** Required. Query interval start (ISO 8601). */
      startTime: string
      /** Optional. Time zone for search times (IANA ID, for example `Europe/Zurich`). Default: the offset of `start_time`, if none then the user's primary time zone. */
      timeZone?: string
    }
    /** Updates an event on the given calendar. */
    mcp__claude_ai_Google_Calendar__update_event: {
      /** Optional. If true, creates or updates a Google Meet URL for the event. Ignored if Meet is disabled. */
      addGoogleMeetUrl?: boolean
      /** Optional. File attachments to add to the event. */
      addedAttachments?: Array<unknown /* $ref #/$defs/Attachment */>
      /** Optional. Deprecated: use `added_attendees` instead. */
      addedAttendeeEmails?: string[]
      /** Optional. Attendees to add to the event. */
      addedAttendees?: Array<unknown /* $ref #/$defs/Attendee */>
      /** Optional. Changes the event to all-day. If set, `start_time`/`end_time` must also be provided. */
      allDay?: boolean
      /** Optional. Whether the event blocks time on the calendar. */
      availability?: "AVAILABILITY_UNSPECIFIED" | "AVAILABILITY_BUSY" | "AVAILABILITY_FREE"
      /** Optional. ID of the calendar containing the event. Email address - can be resolved using `list_calendars`. Default: primary calendar. */
      calendarId?: string
      /** Optional. New color of the event. For a list of color IDs, refer to the documentation of the Event resource. */
      colorId?: string
      /** Optional. New description. Can contain HTML. */
      description?: string
      /** Optional. New end time (ISO 8601). */
      endTime?: string
      /** Required. Event ID. Can be resolved using `list_events` or `search_events`. */
      eventId: string
      /** Optional. Allows attaching an existing Google Meet URL or meeting ID to the event. Overrides the value of `addGoogleMeetUrl`. */
      googleMeetUrl?: string
      /** Optional. Guest permission settings for this event. */
      guestPermissions?: unknown /* $ref #/$defs/GuestPermissions */
      /** Optional. New location. */
      location?: string
      /** Optional. Email notification to send for this event update. Default: `ALL`. */
      notificationLevel?: "NOTIFICATION_LEVEL_UNSPECIFIED" | "NONE" | "EXTERNAL_ONLY" | "ALL"
      /** Optional. If set, replaces all existing reminders for the event. */
      overrideReminders?: Array<unknown /* $ref #/$defs/Reminder */>
      /** Optional. File attachments to remove from the event. */
      removedAttachmentFileUrls?: string[]
      /** Optional. The attendees of the event to remove, as email addresses. */
      removedAttendeeEmails?: string[]
      /** Optional. New start time (ISO 8601). Preserves duration if updating only start. */
      startTime?: string
      /** Optional. New title. */
      summary?: string
      /** Optional. IANA Time Zone Database name (for example, `America/Los_Angeles`). Default: the user's primary time zone. Overrides offsets in `start_time` and `end_time`. */
      timeZone?: string
      /** Optional. New visibility of the event. Possible values are: - `default` - Uses the default visibility for events on the calendar. Default value. - `public` - Event details are visible to all readers of the calendar. - `private` - The event is private and only event attendees may view event details. */
      visibility?: string
    }
    /** List all available agents with their names, descriptions, and modes (primary/subagent) */
    mcp__opencode__opencode_agent_list: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Ask OpenCode a question in one step. Creates a new session, sends your prompt, and returns the AI response. This is the easiest way to interact with OpenCode. */
    mcp__opencode__opencode_ask: {
      /** The question or instruction to send */
      prompt: string
      /** Optional title for the session */
      title?: string
      /** Provider ID (e.g. 'anthropic') */
      providerID?: string
      /** Model ID (e.g. 'claude-3-5-sonnet-20241022') */
      modelID?: string
      /** Model variant (e.g. 'fast', 'smart') */
      variant?: string
      /** Agent to use (e.g. 'build', 'plan') */
      agent?: string
      /** Optional system prompt override */
      system?: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Set authentication credentials for a provider (e.g. API key). Credentials are stored globally and shared across all projects. */
    mcp__opencode__opencode_auth_set: {
      /** Provider ID (e.g. 'anthropic') */
      providerId: string
      /** Auth type (e.g. 'api') */
      type: string
      /** API key or credential value */
      key: string
    }
    /** Get a compact cached progress report for a session. Much cheaper than opencode_conversation or opencode_wait — returns status, todos, and file counts in a single call. Use this to monitor sessions launched with opencode_fire. */
    mcp__opencode__opencode_check: {
      /** Session ID to check */
      sessionId: string
      /** If true, include the last message text (default: false) */
      detailed?: boolean
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Execute a slash command in a session (e.g. /init, /undo, /redo) */
    mcp__opencode__opencode_command_execute: {
      /** Session ID */
      sessionId: string
      /** The slash command to execute (e.g. 'init', 'undo') */
      command: string
      /** Arguments for the command */
      arguments?: string
      /** Agent to use */
      agent?: string
      /** Provider ID */
      providerID?: string
      /** Model ID */
      modelID?: string
      /** Model variant */
      variant?: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** List all available commands (built-in and custom slash commands) */
    mcp__opencode__opencode_command_list: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Get the current opencode configuration */
    mcp__opencode__opencode_config_get: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** List all configured providers and their default models */
    mcp__opencode__opencode_config_providers: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Update the opencode configuration. Pass a partial config object with fields to update. */
    mcp__opencode__opencode_config_update: {
      /** Partial config object with fields to update */
      config: {}
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Get full project context in one call: current project, path, VCS info, config, and available agents. Useful to understand the current state before starting work. */
    mcp__opencode__opencode_context: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Get the full conversation history of a session, formatted for easy reading. Shows all messages with their roles and content. */
    mcp__opencode__opencode_conversation: {
      /** Session ID */
      sessionId: string
      /** Max messages to return (default: all) */
      limit?: number
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Poll for recent events from the OpenCode server. Collects events for the specified duration and returns them. Useful for monitoring session activity, deployments, and system changes. */
    mcp__opencode__opencode_events_poll: {
      /** How long to collect events in milliseconds (default: 3000, max: 30000) */
      durationMs?: number
      /** Maximum number of events to collect (default: 50) */
      maxEvents?: number
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** List files and directories at a path */
    mcp__opencode__opencode_file_list: {
      /** Path to list (defaults to project root) */
      path?: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Read the content of a file */
    mcp__opencode__opencode_file_read: {
      /** File path to read */
      path: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Get status for tracked files (VCS changes: modified, added, deleted, etc.) */
    mcp__opencode__opencode_file_status: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Find files and directories by name (fuzzy match) */
    mcp__opencode__opencode_find_file: {
      /** Search string for file/directory names */
      query: string
      /** Limit results to 'file' or 'directory' */
      type?: "file" | "directory"
      /** Override the project root for the search */
      searchDirectory?: string
      /** Max number of results (1-200) */
      limit?: number
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Find workspace symbols by name (functions, classes, variables, etc.) */
    mcp__opencode__opencode_find_symbol: {
      /** Symbol name to search for */
      query: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Search for text patterns in project files (regex supported). Returns file paths, line numbers, and matching lines. */
    mcp__opencode__opencode_find_text: {
      /** Text or regex pattern to search for in files */
      pattern: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Fire-and-forget: send a task to OpenCode and return immediately. OpenCode works autonomously in the background. Use `opencode_check` to check progress anytime. Best for long-running tasks when you want to do other work in parallel. */
    mcp__opencode__opencode_fire: {
      /** The task or instruction to send */
      prompt: string
      /** Existing session ID to continue (omit to create a new session) */
      sessionId?: string
      /** Session title (only for new sessions) */
      title?: string
      /** Provider ID (e.g. 'anthropic') */
      providerID?: string
      /** Model ID (e.g. 'claude-opus-4-6') */
      modelID?: string
      /** Model variant */
      variant?: string
      /** Agent to use */
      agent?: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Get the status of configured formatters */
    mcp__opencode__opencode_formatter_status: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Check server health and version */
    mcp__opencode__opencode_health: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Dispose the current opencode instance (shuts it down). WARNING: This is destructive and will terminate the server. */
    mcp__opencode__opencode_instance_dispose: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Write a log entry to the opencode server */
    mcp__opencode__opencode_log: {
      /** Service name for the log entry */
      service: string
      /** Log level */
      level: "debug" | "info" | "warn" | "error"
      /** Log message */
      message: string
      /** Extra data to include in the log entry */
      extra?: {}
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Get the status of LSP (Language Server Protocol) servers */
    mcp__opencode__opencode_lsp_status: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Add an MCP server dynamically to opencode */
    mcp__opencode__opencode_mcp_add: {
      /** Name for the MCP server */
      name: string
      /** MCP server configuration object */
      config: {}
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Get the status of all MCP servers configured in opencode */
    mcp__opencode__opencode_mcp_status: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Get details of a specific message in a session */
    mcp__opencode__opencode_message_get: {
      /** Session ID */
      sessionId: string
      /** Message ID */
      messageId: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** List all messages in a session with formatted output showing roles and content */
    mcp__opencode__opencode_message_list: {
      /** Session ID */
      sessionId: string
      /** Maximum number of messages to return */
      limit?: number
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Send a prompt message to a session and wait for the AI response. Use parts to send text, and optionally specify a model. */
    mcp__opencode__opencode_message_send: {
      /** Session ID */
      sessionId: string
      /** The text message to send */
      text: string
      /** Provider ID (e.g. 'anthropic') */
      providerID?: string
      /** Model ID (e.g. 'claude-3-5-sonnet-20241022') */
      modelID?: string
      /** Model variant (e.g. 'fast', 'smart') */
      variant?: string
      /** Agent to use */
      agent?: string
      /** If true, inject context without triggering AI response (useful for plugins) */
      noReply?: boolean
      /** System prompt override */
      system?: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Send a prompt message asynchronously (fire-and-forget, does not wait for response). Use opencode_wait to poll for completion. */
    mcp__opencode__opencode_message_send_async: {
      /** Session ID */
      sessionId: string
      /** The text message to send */
      text: string
      /** Provider ID (e.g. 'anthropic') */
      providerID?: string
      /** Model ID (e.g. 'claude-3-5-sonnet-20241022') */
      modelID?: string
      /** Model variant (e.g. 'fast', 'smart') */
      variant?: string
      /** Agent to use */
      agent?: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Get the current working path of the opencode server */
    mcp__opencode__opencode_path_get: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** List all pending permission requests across all sessions. When a session is blocked waiting for approval (e.g. to run a shell command or access a file outside the project), it appears here. Respond with `opencode_session_permission`. */
    mcp__opencode__opencode_permission_list: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Get the current active project */
    mcp__opencode__opencode_project_current: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Initialize or open a project directory to host an independent OpenCode session. Use this to create new empty folders, or to explicitly open preexisting projects on the host machine for parallel code generation workloads. */
    mcp__opencode__opencode_project_init: {
      /** The absolute file path where the project directory is located or should be created. */
      path: string
    }
    /** List all projects known to the opencode server */
    mcp__opencode__opencode_project_list: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Get available authentication methods for all providers */
    mcp__opencode__opencode_provider_auth_methods: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** List all configured providers with their connection status. Returns a compact summary — use opencode_provider_models to see models for a specific provider. */
    mcp__opencode__opencode_provider_list: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** List available models for a specific provider. Call opencode_provider_list first to see provider IDs. */
    mcp__opencode__opencode_provider_models: {
      /** Provider ID (e.g. 'anthropic', 'openrouter', 'google') */
      providerId: string
      /** Max models to show (default 30). Use 0 for all. */
      limit?: number
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Start OAuth authorization for a provider */
    mcp__opencode__opencode_provider_oauth_authorize: {
      /** Provider ID to authorize */
      providerId: string
    }
    /** Handle OAuth callback for a provider */
    mcp__opencode__opencode_provider_oauth_callback: {
      /** Provider ID */
      providerId: string
      /** OAuth callback data */
      callbackData: {}
    }
    /** Quick-test whether a provider is working. Creates a temporary session, sends a trivial prompt, checks the response, and cleans up. Great for debugging auth issues. */
    mcp__opencode__opencode_provider_test: {
      /** Provider ID to test (e.g. 'anthropic', 'openrouter') */
      providerId: string
      /** Specific model ID to test. If omitted, uses provider default. */
      modelID?: string
      /** Model variant */
      variant?: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Send a follow-up message to an existing session. Use this to continue a conversation started with opencode_ask or opencode_session_create. */
    mcp__opencode__opencode_reply: {
      /** Session ID to reply in */
      sessionId: string
      /** The follow-up message */
      prompt: string
      /** Provider ID */
      providerID?: string
      /** Model ID */
      modelID?: string
      /** Model variant */
      variant?: string
      /** Agent to use */
      agent?: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Get a formatted summary of all file changes made in a session. Shows diffs in a readable format. */
    mcp__opencode__opencode_review_changes: {
      /** Session ID */
      sessionId: string
      /** Specific message ID to get diff for */
      messageID?: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Send a task to OpenCode and wait for completion. Combines session creation, async prompt, and polling into a single tool call. Use this instead of the manual opencode_message_send_async + opencode_wait pattern. */
    mcp__opencode__opencode_run: {
      /** The task or instruction to send */
      prompt: string
      /** Existing session ID to continue (omit to create a new session) */
      sessionId?: string
      /** Session title (only for new sessions) */
      title?: string
      /** Provider ID (e.g. 'anthropic') */
      providerID?: string
      /** Model ID (e.g. 'claude-opus-4-6') */
      modelID?: string
      /** Model variant */
      variant?: string
      /** Agent to use */
      agent?: string
      /** Max seconds to wait for completion (default: 600 = 10 minutes) */
      maxDurationSeconds?: number
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Abort a running session */
    mcp__opencode__opencode_session_abort: {
      /** Session ID to abort */
      id: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Get child sessions of a session */
    mcp__opencode__opencode_session_children: {
      /** Parent session ID */
      id: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Create a new session. Optionally provide a parentID to create a child session, and a title. */
    mcp__opencode__opencode_session_create: {
      /** Parent session ID */
      parentID?: string
      /** Session title */
      title?: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Delete a session and all its data */
    mcp__opencode__opencode_session_delete: {
      /** Session ID to delete */
      id: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Get the diff for a session, optionally for a specific message */
    mcp__opencode__opencode_session_diff: {
      /** Session ID */
      id: string
      /** Message ID (optional) */
      messageID?: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Fork an existing session, optionally at a specific message */
    mcp__opencode__opencode_session_fork: {
      /** Session ID to fork */
      id: string
      /** Message ID to fork at (optional) */
      messageID?: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Get details of a specific session by ID */
    mcp__opencode__opencode_session_get: {
      /** Session ID */
      id: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Analyze the app and create AGENTS.md for a session. NOTE: This is a long-running operation that may take 30-60+ seconds depending on project size. */
    mcp__opencode__opencode_session_init: {
      /** Session ID */
      id: string
      /** Message ID */
      messageID: string
      /** Provider ID (e.g. 'anthropic') */
      providerID: string
      /** Model ID (e.g. 'claude-3-5-sonnet-20241022') */
      modelID: string
      /** Model variant (e.g. 'fast', 'smart') */
      variant?: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** List all sessions */
    mcp__opencode__opencode_session_list: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Respond to a permission request in a session. Use `opencode_permission_list` to see pending requests. Reply values: 'once' (approve this request only), 'always' (approve this + future matching requests for this session), 'reject' (deny the request). */
    mcp__opencode__opencode_session_permission: {
      /** Session ID */
      id: string
      /** Permission request ID */
      permissionID: string
      /** Response to the permission request: 'once' to approve once, 'always' to auto-approve matching future requests, 'reject' to deny */
      reply: "once" | "always" | "reject"
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Revert a message in a session */
    mcp__opencode__opencode_session_revert: {
      /** Session ID */
      id: string
      /** Message ID to revert */
      messageID: string
      /** Part ID to revert (optional) */
      partID?: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Search sessions by keyword in title. Useful for finding a specific session among many. */
    mcp__opencode__opencode_session_search: {
      /** Search keyword (case-insensitive match on session title) */
      query: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Share a session publicly */
    mcp__opencode__opencode_session_share: {
      /** Session ID to share */
      id: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Get status for all sessions (running, idle, etc.) */
    mcp__opencode__opencode_session_status: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Summarize a session using a specified model. NOTE: This is a long-running operation that may take 30-60+ seconds. */
    mcp__opencode__opencode_session_summarize: {
      /** Session ID */
      id: string
      /** Provider ID (e.g. 'anthropic') */
      providerID: string
      /** Model ID (e.g. 'claude-3-5-sonnet-20241022') */
      modelID: string
      /** Model variant (e.g. 'fast', 'smart') */
      variant?: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Get the todo list for a session */
    mcp__opencode__opencode_session_todo: {
      /** Session ID */
      id: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Restore all reverted messages in a session */
    mcp__opencode__opencode_session_unrevert: {
      /** Session ID */
      id: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Unshare a previously shared session */
    mcp__opencode__opencode_session_unshare: {
      /** Session ID to unshare */
      id: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Update session properties (e.g. title) */
    mcp__opencode__opencode_session_update: {
      /** Session ID */
      id: string
      /** New title for the session */
      title?: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Get a quick overview of all sessions with their titles and status. Useful to find which session to continue working in. */
    mcp__opencode__opencode_sessions_overview: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Check OpenCode status, provider configuration, and optionally initialize a project directory. Use this as the first step when starting work — it tells you what is ready and what still needs configuration. */
    mcp__opencode__opencode_setup: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Run a shell command through the opencode session */
    mcp__opencode__opencode_shell_execute: {
      /** Session ID */
      sessionId: string
      /** Shell command to execute */
      command: string
      /** Agent to use for the shell command */
      agent: string
      /** Provider ID */
      providerID?: string
      /** Model ID */
      modelID?: string
      /** Model variant */
      variant?: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Get a quick status dashboard: server health, provider count, session count, and VCS info. Lighter than opencode_setup — good for at-a-glance checks. */
    mcp__opencode__opencode_status: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** List all available tool IDs that the LLM can use (experimental) */
    mcp__opencode__opencode_tool_ids: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** List tools with JSON schemas for a given provider and model (experimental) */
    mcp__opencode__opencode_tool_list: {
      /** Provider ID */
      provider: string
      /** Model ID */
      model: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Append text to the TUI's prompt input field */
    mcp__opencode__opencode_tui_append_prompt: {
      /** Text to append to the prompt */
      text: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Clear the current prompt text in the TUI */
    mcp__opencode__opencode_tui_clear_prompt: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Execute a slash command through the TUI (e.g. '/init', '/undo') */
    mcp__opencode__opencode_tui_execute_command: {
      /** Command to execute (e.g. '/init') */
      command: string
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Open the help dialog in the TUI */
    mcp__opencode__opencode_tui_open_help: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Open the model selector in the TUI */
    mcp__opencode__opencode_tui_open_models: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Open the session selector in the TUI */
    mcp__opencode__opencode_tui_open_sessions: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Open the theme selector in the TUI */
    mcp__opencode__opencode_tui_open_themes: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Show a toast notification in the TUI */
    mcp__opencode__opencode_tui_show_toast: {
      /** Toast message text */
      message: string
      /** Optional toast title */
      title?: string
      /** Toast variant (default: info) */
      variant?: "info" | "success" | "warning" | "error"
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Submit the current prompt in the TUI (equivalent to pressing Enter) */
    mcp__opencode__opencode_tui_submit_prompt: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Get VCS (version control) info for the current project (branch, remote, status) */
    mcp__opencode__opencode_vcs_info: {
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
    /** Poll a session until it finishes processing. Use after opencode_message_send_async to wait for the AI to complete its response. Sends progress notifications while waiting. If timeout is reached, returns a progress report (not an error). For long tasks, consider using opencode_session_todo to check progress instead of blocking. */
    mcp__opencode__opencode_wait: {
      /** Session ID to wait on */
      sessionId: string
      /** Max seconds to wait (default: 120). Set higher (300-600) for complex tasks. */
      timeoutSeconds?: number
      /** Polling interval in ms (default: 2000) */
      pollIntervalMs?: number
      /** Absolute path to the project directory. When provided, the request targets that project. If omitted, the OpenCode server uses its own working directory. */
      directory?: string
    }
  }
}
