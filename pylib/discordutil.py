# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# discord_aiagent.discordutil
import asyncio
import datetime

import discord
from discord import abc as discord_abc
from discord.ext import commands

import structlog

logger = structlog.get_logger(__name__)


def get_formatted_sysmsg(sysmsg_template: str, discord_user_id: str | int, last_execution_time: float = 0.0) -> str:
    '''
    Formats the system message template with dynamic values.
    '''
    current_time_date_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')
    
    # Format last execution time if provided
    last_execution_str = ''
    if last_execution_time > 0.0:
        last_execution_dt = datetime.datetime.fromtimestamp(last_execution_time)
        last_execution_str = last_execution_dt.strftime('%Y-%m-%d %H:%M:%S %Z')

    # Prepare the data for formatting
    format_data = {
        'DISCORD_ID': str(discord_user_id),
        'CURRENT_TIME_DATE': current_time_date_str,
        'LAST_EXECUTION_TIME': str(last_execution_time) if last_execution_time > 0.0 else '0',
        'LAST_EXECUTION_DATE': last_execution_str
    }

    formatted_message = sysmsg_template.format_map(format_data)
    logger.debug('Formatted system message', sysmsg=formatted_message)

    return formatted_message


async def send_long_message(sendable: commands.Context | discord.Interaction | discord_abc.Messageable, content: str, followup: bool = False, ephemeral: bool = False):
    '''
    Sends a message, splitting it into multiple parts if it exceeds Discord's limit.
    Handles both Context and Interaction objects.
    '''
    max_length = 2000
    start = 0
    # Ensure content is a string, even if it's None or another type unexpectedly
    content_str = str(content) if content is not None else ''
    if not content_str.strip():  # Don't try to send empty messages
        logger.debug('Attempted to send empty message.')
        return

    first_chunk = True
    while start < len(content_str):
        end = start + max_length
        chunk = content_str[start:end]
        try:
            if isinstance(sendable, commands.Context):
                await sendable.send(chunk)
            elif isinstance(sendable, discord.Interaction):
                use_followup = followup or start > 0 or sendable.response.is_done()
                can_be_ephemeral = (use_followup and first_chunk) or (not use_followup and len(content_str) <= max_length)
                send_ephemeral = ephemeral and can_be_ephemeral

                if use_followup:
                    if not sendable.response.is_done():
                        # Interaction likely wasn't deferred or followup wasn't requested initially.
                        # Use original response if possible, otherwise log warning.
                        # This path is less common if defer() is used consistently.
                        try:
                            await sendable.response.send_message(chunk, ephemeral=send_ephemeral)
                            logger.debug('Sent initial interaction response (unexpected path after deferral).')
                        except discord.InteractionResponded:
                            logger.warning('Interaction already responded, but followup wasn\'t triggered correctly. Sending via followup.')
                            await sendable.followup.send(chunk, ephemeral=send_ephemeral)  # Fallback to followup
                    else:
                        await sendable.followup.send(chunk, ephemeral=send_ephemeral)
                else:
                     # Initial response for non-deferred interaction or first message.
                     await sendable.response.send_message(chunk, ephemeral=send_ephemeral)
            elif isinstance(sendable, discord_abc.Messageable):
                # DMs don't support ephemeral or followups in the same way as interactions
                # Just send directly to the channel
                await sendable.send(chunk)
                # Reset ephemeral flag for logging clarity if it was passed for a DM
                ephemeral = False

        except discord.HTTPException as e:
            logger.error(f'Failed to send message chunk: {e}', chunk_start=start, sendable_type=type(sendable), ephemeral=ephemeral, status=e.status, code=e.code)
            # Try to inform user ephemerally if possible
            error_msg = f'Error sending part of the message (Code: {e.code}). Please check channel permissions or message content.'
            try:
                if isinstance(sendable, discord.Interaction):
                    # Use followup if possible, it's more reliable after potential initial errors
                    if sendable.response.is_done():
                        await sendable.followup.send(error_msg, ephemeral=True)
                    else:
                        # Avoid double-responding if initial send failed
                        pass  # await sendable.response.send_message(error_msg, ephemeral=True)  # Risky if initial failed
                # else: await sendable.send(error_msg)  # Avoid sending public errors if interaction failed
            except Exception:
                pass  # Avoid error loops
            break  # Stop sending further chunks on error

        start = end
        first_chunk = False
        if start < len(content_str):  # Only sleep if more chunks are coming
            await asyncio.sleep(0.2)


async def handle_attachments(attachments: list[discord.Attachment], message) -> str:
    '''
    Handles attachments in a message, sending them to the specified sendable context.
    '''
    attachment_texts = []
    for attachment in attachments:
        try:
            # Basic info
            attachment_info = f'\n\n--- Attachment: {attachment.filename} ({attachment.content_type}, {attachment.size} bytes) ---\n'

            # Placeholder for file type registration and preprocessing
            # Example:
            # registered_handler = file_handlers.get(attachment.content_type)
            # if registered_handler:
            #     content = await registered_handler(attachment)
            # elif attachment.content_type.startswith('image/'):
            #     # Placeholder for image-to-text model
            #     content = f'[Content of image {attachment.filename} - requires image model]'
            # elif attachment.filename.endswith('.xlsx'):
            #     # Placeholder for Excel to CSV/text conversion
            #     content = f'[Content of Excel file {attachment.filename} - requires processing]'
            # elif attachment.content_type.startswith('text/'):
            #     # Default: Read text-based files directly
            #     content_bytes = await attachment.read()
            #     # Attempt decoding, fall back to representing as bytes if fails
            #     try:
            #         content = content_bytes.decode('utf-8')  # Or try other common encodings
            #     except UnicodeDecodeError:
            #         logger.warning(f'Could not decode attachment {attachment.filename} as UTF-8.', attachment_id=attachment.id)
            #         content = f'[Binary content of {attachment.filename}]'
            # else:
            #     # Default for unknown types
            #     content = f'[Unsupported attachment type: {attachment.content_type}]'

            # Default behavior: Read text files, provide placeholder for others
            if attachment.content_type and attachment.content_type.startswith('text/'):
                try:
                    content_bytes = await attachment.read()
                    content = content_bytes.decode('utf-8')
                    # Optional: Truncate large files
                    max_len = 5000  # Example limit
                    if len(content) > max_len:
                        content = content[:max_len] + '\n[... content truncated ...]'
                except UnicodeDecodeError:
                    logger.warning(f'Could not decode text attachment {attachment.filename} as UTF-8.', attachment_id=attachment.id)
                    content = f'[Could not decode content of {attachment.filename}]'
                except discord.HTTPException as e:
                    logger.error(f'Failed to read attachment {attachment.filename}: {e}', attachment_id=attachment.id)
                    content = f'[Error reading attachment {attachment.filename}]'
            else:
                # Placeholder for non-text files (images, pdf, excel etc.)
                content = f'[Content of non-text file: {attachment.filename} ({attachment.content_type}) - requires specific processing]'

            attachment_texts.append(attachment_info + content)

        except Exception as e:
            logger.exception(f'Error processing attachment {attachment.filename}', attachment_id=attachment.id)
            attachment_texts.append(f'\n\n[Error processing attachment: {attachment.filename}]')

    # Prepend attachment contents to the user's message
    processed_message = ''.join(attachment_texts) + '\n\n--- User Message ---\n' + message
    logger.debug(f'Added content from {len(attachments)} attachments to the message.')
    return processed_message
