# Copyright (c) 2021, Summarizers. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from experiments.models.bart import BartForSeq2SeqLM

model = BartForSeq2SeqLM(
    cfg_path="../configs",
    cfg_name="bart_for_text_summarization",
)

# model.fit(
#     train_dataloader=YOUR_TRAIN_DATALOADER
#     val_dataloader=YOUR_VALIDATION_DATALOADER
# )
