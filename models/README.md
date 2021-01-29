# Generative Models

## Generator and Discriminator

First of all, we thank following repositories for their work on high-quality image synthesis

- [PGGAN](https://github.com/tkarras/progressive_growing_of_gans)
- [StyleGAN](https://github.com/NVlabs/stylegan)
- [StyleGAN2](https://github.com/NVlabs/stylegan2)

Pre-trained tensorflow weights (either officially released or trained by ourselves) can be found from following links. Please download them and save to folder `pretrain/tensorflow/` before using.

**NOTE:** The officially released models are simply mirrored by us from the above three repositories, just in case they are not available from the official links.

| PGGAN Official | | | |
| :-- | :-- | :-- | :-- |
| *Face*
| [celebahq-1024x1024](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ERkthZuF1rBCrJRURQ5M1W8BbsfT5gFF-TGbuxCAuUJXPQ?e=uKYyQ1&download=1)
| *Indoor Scene*
| [bedroom-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EZeWkI9pbUZDqZAzEUDjlSwB5nDZhe94vmmg4G5QSKGy7A?e=5RhTOo&download=1)       | [livingroom-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EbHv-4YvGYJJl6i4zH8s25kBqpA1RG-YZbAvp2PSc5CtRA?e=SnSk49&download=1) | [diningroom-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/Ee2LUJ6fectMiFDYYrZiA1sBD5q4j_FBC8xzH2Z6GSb-JQ?e=pxhVrt&download=1)  | [kitchen-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ERTDgXhOqJZPlM72bULyKsgBu7nABHvmCBIbwvASzKruvg?e=lIrB34&download=1)
| *Outdoor Scene*
| [churchoutdoor-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EfPAIPVXbYxIn0KQ5IzCJxYBfEG4nP1p7D3MK-N24HLzow?e=za16Z1&download=1) | [tower-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EXZGTFQX8gNPgwvCGWKmiIwBxGgU4UTIQy1wezKnpAADMg?e=KUp4hJ&download=1)      | [bridge-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EXba4rsRrcZDg_6SQk-vClMBmqesihPHY6fne5oobKLHhg?e=9Gk1v3&download=1)
| *Other Scene*
| [restaurant-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/Eb0vWXX-n5BLkf9jL61ekfwBxHDpFxVLq9igSYJyQ3x5FQ?e=EuqMTU&download=1)    | [classroom-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EW5vDIwjV6dPsfK_szdVnTABVsd11xvJ_O6-ReVeQsvtQA?e=dls0Jd&download=1)  | [conferenceroom-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/Eb1kh2L4ayxFjXQL2y34yEkBb_eZ9pcSXnY3ivnvCdeknA?e=wPATWN&download=1)
| *Animal*
| [person-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EbILxVQAbd9HsjxXwiOX2PABWHvmIsgrdwmvF0PPQl8_Xw?e=799btl&download=1)        | [cat-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/Ebr89QFQnRJHv-OQ7IMgu-YBG02kswtRukk-9ylUqY8bGQ?e=ioo5m4&download=1)        | [dog-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EeC5DITcQUNFkBPaVFnS4-YBOpFaVb_5agq_vkPG_aFvlg?e=rnq8Rw&download=1)         | [bird-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EbvqTPl0ru5MicpQbuIePtgBSwDbzef23TgcrCNcFX5A-A?e=jMRaqB&download=1)
| [horse-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EfsJ0u6ZhDhHvleYRd5OCYABCd6Q6uqU1l-AM_C-Cot5_g?e=Fqmudf&download=1)         | [sheep-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EaIy20hZi5pHkVZhO7p38OoBrjInx6UAFzwAMtG_fcnUCg?e=A6ax03&download=1)      | [cow-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ETcm1hzw7M5Mmbi1vHNAA1sBNZcCwXr1Y_y-nwVqEcNHKQ?e=IE0Cu0&download=1)
| *Transportation*
| [car-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/Ec6LMgv8jSpJo9MHv39boLkBR6zqrnK_XCjrJdDDoIjfTg?e=HKRIet&download=1)           | [bicycle-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EW9S1pnWUXtAuHLRcFeoHmYB0vmHwdf6ipxMIPOzxQnOaw?e=pbEBXp&download=1)    | [motorbike-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ESiSYhItLfZKnsuW1bA6XPMBHt9Um3p2WvEknOndLgNLtw?e=uVCCIx&download=1)   | [bus-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EY996zZOBAFDip6W18m0OY0BERSAl_CoVJt0mCUNod2bBg?e=Mt8Qgg&download=1)
| [train-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EeqoMneXJ6hKkuVoKTvgfG8Bbn7yx6FGByzzpF8avQ5ecw?e=7b0rb1&download=1)         | [boat-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ESL-wYbgG2NMmEfjNqVe6DcB0wHkx-GeFsWWnmnhK6DL6Q?e=yVwAUW&download=1)       | [airplane-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EYsnUkaD7kZNjHLCeTcEEaYBjNPO6_wra4Erlh6SMCs3eQ?e=wvRubM&download=1)
| *Furniture*
| [bottle-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EUjdW87xSmVCumxRS0E6OXUB67wxAjappdW4XHvbOx3UgA?e=GT46ho&download=1)        | [chair-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EYgI1WgBJ5NPomn9BedMkRkBTaKcQOIaoGWQg-oe-eVN8g?e=42YuAT&download=1)      | [pottedplant-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EdxtDuYh_31Fpc6TA5ZtATQB2b2IqnwG0z4NzDzYfNHSOw?e=QV213z&download=1) | [tvmonitor-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EfXmQpZbx35KuZkuZO_C_qsBYaFnnP6Cq9al4NI6-lrqLQ?e=Y2EEy8&download=1)
| [diningtable-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EW0UuwPB3pZNh5jTUDHEl24BaIiqvcB-_9k1TpX3nRFhvw?e=xw7CYQ&download=1)   | [sofa-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EYHZdQA2DJBGjetN2agEnEEBicfxWbzMMON5wlgNDc5AFw?e=JsTzLG&download=1)

| StyleGAN Official | | | |
| :-- | :--: | :--: | :--: |
| Model (Dataset) | Training Samples | Training Duration (K Images) | FID
| [ffhq-1024x1024](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ERrWZh5VmmFPkMPBqWyj88kBWevxE-_mELOo9toH8LEK9A?e=W7DS8j&download=1)     |    70,000 | 25,000 | 4.40
| [celebahq-1024x1024](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EZLLovAsvOBNhRrElHONwgYBsRy1QRc_kIOGpvHhEUar3w?e=ORRaR3&download=1) |    30,000 | 25,000 | 5.06
| [bedroom-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EZZtgyznB4xNm9gmLyOBHpcB-ohvHMmKZmZx6Tfx9_o8HA?e=gV0ZXi&download=1)    | 3,033,042 | 70,000 | 2.65
| [cat-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ETHKOTaZPEBNkVoM-zJgDVQBciac5PZFsbVw8raYycDTlA?e=R8oxiP&download=1)        | 1,657,266 | 70,000 | 8.53
| [car-512x384](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EU-MbMEy8IlGtnjXv-dBF7cBZTJoGqG4le-NvyrURvS4Eg?e=IxDwat&download=1)        | 5,520,756 | 46,000 | 3.27

| StyleGAN Ours | | | |
| :-- | :--: | :--: | :--: |
| Model (Dataset) | Training Samples | Training Duration (K Images) | FID
| *MNIST*
| [mnist-32x32](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EZJyjUFCiJZKoclPoPb1lXIBIVCHc91OAPRvkTlejawWEw?e=PW4QYM&download=1)                         |      60,000 | 20,000 |
| [mnist_cond-32x32](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ERn5TgGMKR1Gvti7cfp0eXYBwubqNMQElB0BIyhCSNsWuQ?e=kJMvGh&download=1) (10 classes)       |      60,000 | 20,000 |
| [mnist_color-32x32](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EZ_KLn-AlW9Bv86KabPce7EBxuEMHFG8vK2jwrCH6GWc5Q?e=oPk7PH&download=1)                   |      60,000 | 20,000 |
| [mnist_color_cond-32x32](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EW6A3XyZX9ZLga44poKVvuIBdtBEZeiRGv-YrTtrxiBbhA?e=W1HbIa&download=1) (10 classes) |      60,000 | 20,000 |
| *SVHN*
| [svhn-32x32](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EUsby2gJrxhKgBwS39oHud0B8Yir9m7vwCPK-sUv_xtDBw?e=j9GiV9&download=1)                          |      73,257 | 30,000 |
| [svhn_cond-32x32](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EVRmEK8dbEZLp9xWv1-elZgBI1SCaE71UIQI2WAdLzFjgw?e=hxEHML&download=1) (10 classes)        |      73,257 | 30,000 |
| *CIFAR*
| [cifar10-32x32](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EcVfVjss6RFDn_HKY1F5fxMBKnMxH-JjXATu0R8bvjFshg?e=3PGqfK&download=1)                       |      50,000 | 30,000 |
| [cifar10_cond-32x32](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EfVv1JP5p55BisYBIvkuX8wBpk1XLOVZRtyXsESzrIZ3Dg?e=0klPxP&download=1) (10 classes)     |      50,000 | 30,000 |
| [cifar100-32x32](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/Ecu7FR0rJtpLruHdovVNPLIBetjvaL5jnqW6iJGeVkQY_Q?e=ZJKOhr&download=1)                      |      50,000 | 30,000 |
| [cifar100_cond-32x32](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ERcB1XflUm9GpL_RErEMYmYBPbPvs9qQ2NE7y6UOVcHtsQ?e=sT884k&download=1) (100 classes)   |      50,000 | 30,000 |
| *Face ("partial" means faces are not fully aligned to center)*
| [celeba_partial-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ET3c5bFnbmlLlsSE5mQCg5IB3NLYwPobGiQOhuEHOaletQ?e=nynbYH&download=1)              |     103,706 | 50,000 |  7.03
| [ffhq-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EQBQjcqqgoNMkFNUjfE69soB5KPWHhSHW_LaOlH9WJ-uHw?e=AByHJO&download=1)                        |      70,000 | 25,000 |  5.70
| [ffhq-512x512](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EUqBKsTPSX5KgBiN6mQ68fgB7pVdk4itrK7Budnxvd9FxA?e=wxVr6q&download=1)                        |      70,000 | 25,000 |  5.15
| *LSUN Indoor Scene*
| [livingroom-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EW3M1ZzNc4REgBuFMD1soLgBQCteWBZdJsH7eCcRfJ-P-Q?e=LyfLj7&download=1)                  |   1,315,802 | 30,000 |  5.16
| [diningroom-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EaZ1XWbU4KNKkD9SBUqtMXcBCq6ywjyeq-_OQ8sCUR6rzQ?e=rjOTcA&download=1)                  |     657,571 | 25,000 |  4.13
| [kitchen-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EZ-3iOBSeqtKlIHWfOC4_-0BfzYwNHPNNYNVho2lkqm_Rw?e=TBAxAS&download=1)                     |   1,000,000 | 30,000 |  5.06
| *LSUN Indoor Scene Mixture*
| [apartment-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EWvK04bleE1DrNO_GbtY4BsBZtqzSWJZ_VtxMkSJiK4QTg?e=WG74Jg&download=1)                   | 4 * 200,000 | 60,000 |  4.18
| *LSUN Outdoor Scene*
| [church-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EcfDRkV7ncNJhJTsfbrli0MBnEPQXJeyNZ2FzS6XeAzKxA?e=Woibfx&download=1)                      |     126,227 | 30,000 |  4.82
| [tower-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EXU65vZbVF5JhdqKWg8x7FkBXp8DCwdqPA26IkSiiKtLqw?e=nEkOQa&download=1)                       |     708,264 | 30,000 |  5.99
| [bridge-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EWyzlQIgxNxOrcOkzb_GewkBqH5GTfKiMV1B27z5QJIJrw?e=6kgyan&download=1)                      |     818,687 | 25,000 |  6.42
| *LSUN Other Scene*
| [restaurant-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EZdxja8kJ8hFgVi4iCKApuoBRJ9HKUdNF53giR9D61V5jQ?e=8B1kLn&download=1)                  |     626,331 | 50,000 |  4.03
| [classroom-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EfgXckBHSfZHsf_FUBXAsl8Btt6X0SRr1O8-FqyNbIaXRw?e=yZ5z8q&download=1)                   |     168,103 | 50,000 | 10.10
| [conferenceroom-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EeOwgtZORopBibIOI022TYIBv1YPVpGy0FLM386olADZOg?e=hzaZzZ&download=1)              |     229,069 | 50,000 |  6.20

| StyleGAN Third-Party | |
| :-- | :--: |
| Model (Dataset) | Source |
| [animeface-512x512](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ERw-zBpwKvpMry3bw-U8B_UBzuTJvN0J5KCzmOHYesPUWA?e=krxWFV&download=1)     | [link](https://www.gwern.net/Faces#portrait-results)
| [animeportrait-512x512](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ETLEcatI3lZGqjQ_RSTKPDQBpg5JdT_b9iI1kd4mxXSHFg?e=WnH527&download=1) | [link](https://www.gwern.net/Faces#portrait-results)
| [artface-512x512](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/Ee1qhGo54-ZMk34goK3Dc8MBXUbQBCzFlfok2KzEgA7CPA?e=kZXdKj&download=1)       | [link](https://github.com/ak9250/stylegan-art)

| StyleGAN2 Official | | | |
| :-- | :--: | :--: | :--: |
| Model (Dataset) | Training Samples | Training Duration (K Images) | FID
| [ffhq-1024x1024](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/Eb053e-OAblEuS7ZnUto8R0Bub4HtnF5nJVoUUeLPA7Kbw?e=Kv3O8m&download=1) |    70,000 |  25,000 | 2.84
| [church-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/Ec88e85Iz_tMs9C2cSY1Bw4BCmVwCXJCFpHpWMkQ7HkcUA?e=ejGLi6&download=1) |   126,227 |  48,000 | 3.86
| [cat-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ERM88hE3pTlHm_8D9OMxwYUB20Yij8IshFwk0F6C2LV2pQ?e=4wmOQ5&download=1)    | 1,657,266 |  88,000 | 6.93
| [horse-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EYdKeNQctlRHlFZX5aUIs2kBicFtS_MwSPSgemPJecVvzw?e=RiEM2H&download=1)  | 2,000,340 | 100,000 | 3.43
| [car-512x384](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EWk22jukftBInK98BnW6hrgBSEPxtvYO4li8EFQhIj28wg?e=VeloK1&download=1)    | 5,520,756 |  57,000 | 2.32

| StyleGAN2 Ours | | | |
| :-- | :--: | :--: | :--: |
| Model (Dataset) | Training Samples | Training Duration (K Images) | FID
| *MNIST*
| [mnist-32x32](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EQebaOQ6-iRIrQF_Zd7mS_cBDg1qADvUpQhlLdBlVpT6DQ?e=nPdQKe&download=1)                          |      60,000 |  20,000 |  3.05
| [mnist_cond-32x32](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EaDww0idnBVIhjzzL2BMBUcBERiH5JWOnYnbANir2s4_6Q?e=iatbaf&download=1) (10 classes)        |      60,000 |  20,000 |  1.71
| [mnist_color-32x32](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EcvBtQsphAVPsODlqBr8AC4B-8RsjnT0W9v1IJh33q_0ZQ?e=tbU6Zn&download=1)                    |      60,000 |  20,000 |  4.21
| [mnist_color_cond-32x32](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EViq_Kn6CbBMlRms2krU6_wBuUpgMeey0ZsNZnEHzRnOSQ?e=F8T8Rw&download=1) (10 classes)  |      60,000 |  20,000 |  1.65
| *SVHN*
| [svhn-32x32](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EVktZCh2pl5HhQA3oI3p39cB7W_djVPCiCWV9gfi3PpPkg?e=S5NdSs&download=1)                           |      73,257 |  30,000 |  7.94
| [svhn_cond-32x32](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EQMgCzQcFeZNvYeKSS3i0SEBheoWmRRACap6cn2t-kL3Ug?e=M74Rjo&download=1) (10 classes)         |      73,257 |  30,000 |  9.34
| *CIFAR*
| [cifar10-32x32](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ET7UFMOhmPxIp_3BBvm4K3cBspT_uuwFxvJfxShFomCwgw?e=6r4l3c&download=1)                        |      50,000 |  10,000 | 16.90
| [cifar10_cond-32x32](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EeG8FzEEBBpHvsXgf-hbCgoBFt0ahv2w2wUcIysfs8Eu3Q?e=kMDL21&download=1) (10 classes)      |      50,000 |  10,000 | 21.41
| [cifar100-32x32](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EbcQkjRrNJNCnOKdKDPVyJ8Bq0T3N_TouUJmCVEQACMBjQ?e=dghEGY&download=1)                       |      50,000 |  10,000 | 23.28
| [cifar100_cond-32x32](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EaobGsBwvwhHmRDY_dKbpX4BvF6lvxlYWZGgNpPcVW6MEQ?e=yJAvOk&download=1) (100 classes)    |      50,000 |  10,000 | 26.32
| *ImageNet*
| [imagenet-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ERsC79nYFplHm__gMneMpfgBzBhYuGJkUn8QGjUnjesteQ?e=s91Phd&download=1)                     |   1,281,167 | 250,000 | 40.92
| [imagenet_cond-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ETmdijcSVaVDvJfnxQTbcBgBPQXJlLOTSuJVA9iMxv7BcQ?e=3V20kZ&download=1) (1000 classes) |   1,281,167 | 250,000 | 29.95
| *Face ("partial" means faces are not fully aligned to center)*
| [celeba_partial-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EbAYjiw0oalPpwYeSX7IUQwBXnKjxprhxfZRxvcr88KGrg?e=GGgHhb&download=1)               |     103,706 |  50,000 |  3.83
| [ffhq-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EWq3JbLyiLdEksMz_EJNQS8BJPFxfH1Y4dmuPOscIucdvA?e=DsY54V&download=1)                         |      70,000 |  25,000 |  5.04
| [ffhq-512x512](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EUE6_abYNcpCvAMgPRSQzGYB7wdBy08VTfcRiqXN227Egw?e=RoKdpO&download=1)                         |      70,000 |  25,000 |  3.69
| *LSUN Indoor Scene*
| [bedroom-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EXgg_IWflUJOlGWlPvAbnvoBOSsSZ6RPj5rYdWrb00heSQ?e=ZT8aUG&download=1)                      |   3,033,042 |  60,000 |  2.93
| [livingroom-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ETfzPoKcE6lJkiOA8nzg3BIB30IkDdVEyiq8eMwNa2TSqQ?e=SPmvSE&download=1)                   |   1,315,802 | 100,000 |  3.48
| [diningroom-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EShU1lT1f25KgZHSrF_Ih5ABES-z1H5Ib4SZwI9-f6PCjg?e=6VwKAb&download=1)                   |     657,571 |  60,000 |  2.69
| [kitchen-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EdNjlhMg9odNpr9HF5Yb3rIBmq5xr7gKbDUDCCNEVlmUFQ?e=JtX1jh&download=1)                      |   2,212,277 | 100,000 |  3.05
| *LSUN Indoor Scene Mixture*
| [apartment-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EeYifSJWiqVOkYB3DQ1rK9QBns33_AvcYXPMAXS9ZYLZow?e=dsoWxc&download=1)                    | 4 * 500,000 | 120,000 |  2.88
| [apartment_cond-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EXtWeIu0ZxtBr_I4iLZ4lnEBCMP1drtTs7BPkW5KNad2kw?e=K0ZnDJ&download=1) (4 classes)   | 4 * 500,000 | 120,000 |  4.14
| *LSUN Outdoor Scene*
| [tower-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EcmUrApBlXlEup-Mb1ykb90BS-vqQhgmBz4OwiXB5WvYxQ?e=U6kcUp&download=1)                        |     708,264 |  60,000 |  3.60
| [bridge-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ERAWfRJa5RNMuz8LLK3romQBr8KV9FQRufMLVq0ZEr_Vsw?e=0MFeKF&download=1)                       |     818,687 |  60,000 |  3.65
| *LSUN Other Scene*
| [restaurant-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EeQ8CuKmBwhHvrMcSH2y7UIBZCRv0EJ-pDsYwcCDFFCT6Q?e=GbO9cM&download=1)                   |     626,331 | 120,000 |  4.48
| [classroom-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EWY75DRKGElJqnPTy89HrvUBC-h8RTs79F90sMHHj6kNMA?e=TXpmSe&download=1)                    |     168,103 |  60,000 |  6.19
| [conferenceroom-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/Eeqviy9fA3tAiwdYR-8Lpx4BSyzGTO2KE-eTF20arpnbWA?e=qWg4aA&download=1)               |     229,069 |  60,000 |  4.41
| *Places*
| [places-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EYtZ5YOghCJHpkUXRH0cT70BexjuHxXCxbhp7HFrBvOihQ?e=pz1cis&download=1)                       |   1,803,460 | 120,000 | 10.16
| [places_cond-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EZ1FCQ7sTH1CqbXWH6-ry5cBb9hJZ-_Zhh4F9VNWWIq1SA?e=bBCCUi&download=1) (365 classes)    |   1,803,460 | 120,000 |  8.04
| *Cityscapes*
| [cityscapes-1024x1024](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ESqj27KGVrtPlHcGhPZXFf4BLWQn53K2Oxtpb1vSe2qojA?e=geZ4D2&download=1)                 |      24,997 |  15,000 |  8.31
| *MIT Streetscapes*
| [streetscapes-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/Eczd669jT2NFjOoR8SKthkMBAjyUzYQXTung2T9sVnymKw?e=mQ1UrB&download=1)                 |     116,235 |  50,000 |  2.81

## Encoder

The new pickle file consists of `(E, G, D, Gs)` in sequence.

| StyleGAN Encoder |
| :-- |
| [styleganinv-ffhq-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EWmmTzOZP5dArApiXnEzz44BL2O8CY6ee0DkNUpAzF6Mww?e=RIJzT5&download=1)
| [styleganinv-bedroom-256x256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EZR48uwT-gNFsIuZ1gOqyIsBN5d_ynWKcckFTvpgtyJr_A?e=Sjf4pb&download=1)

## Perceptural Model (VGG16 from Keras)

Please download the pre-trained [VGG16 model](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5), and save it to the folder `pretrain/`. It is currently a Keras model and will be automatically converted to PyTorch version when used for the first time.

## Training Datasets

Datasets used in this work can be found from following repositories

- [MNIST](http://yann.lecun.com/exdb/mnist/) (60,000 training samples and 10,000 test samples on 10 digital numbers)
- [SVHN](http://ufldl.stanford.edu/housenumbers/) (73,257 training samples, 26,032 testing samples, and 531,131 additional samples on 10 digital numbers)
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) (50,000 training samples and 10,000 test samples on 10 classes)
- [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) (50,000 training samples and 10,000 test samples on 100 classes)
- [ImageNet](http://www.image-net.org/) (1,281,167 training samples, 50,000 validation samples, and 100,100 testing samples on 1000 classes)
- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (202,599 samples from 10,177 identities, with 5 landmarks and 40 binary facial attributes)
- [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans) (30,000 samples)
- [FF-HQ](https://github.com/NVlabs/ffhq-dataset) (70,000 samples)
- [LSUN](https://github.com/fyu/lsun) (see statistical information below)
- [Places](http://places2.csail.mit.edu/) (around 1.8M training samples covering 365 classes)
- [Cityscapes](https://www.cityscapes-dataset.com/) (2,975 training samples, 19998 extra training samples (one broken), 500 validation samples, and 1,525 test samples)
- [Streetscapes](http://streetscore.media.mit.edu/data.html)

Statistical information of LSUN dataset is summarized as follows:

| LSUN Datasets Stats | | |
| :-- | :--: | :--: |
| Name | Number of Samples | Size |
| *Scenes*
| bedroom (train)        |  3,033,042 |  43G
| bridge (train)         |    818,687 |  15G
| churchoutdoor (train)  |    126,227 |   2G
| classroom (train)      |    168,103 |   3G
| conferenceroom (train) |    229,069 |   4G
| diningroom (train)     |    657,571 |  11G
| kitchen (train)        |  2,212,277 |  33G
| livingroom (train)     |  1,315,802 |  21G
| restaurant (train)     |    626,331 |  13G
| tower (train)          |    708,264 |  11G
| *Objects*
| airplane               |  1,530,696 |  34G
| bicycle                |  3,347,211 | 129G
| bird                   |  2,310,362 |  65G
| boat                   |  2,651,165 |  86G
| bottle                 |  3,202,760 |  64G
| bus                    |    695,891 |  24G
| car                    |  5,520,756 | 173G
| cat                    |  1,657,266 |  42G
| chair                  |  5,037,807 | 116G
| cow                    |    377,379 |  15G
| diningtable            |  1,537,123 |  48G
| dog                    |  5,054,817 | 145G
| horse                  |  2,000,340 |  69G
| motorbike              |  1,194,101 |  42G
| person                 | 18,890,816 | 477G
| pottedplant            |  1,104,859 |  43G
| sheep                  |    418,983 |  18G
| sofa                   |  2,365,870 |  56G
| train                  |  1,148,020 |  43G
| tvmonitor              |  2,463,284 |  46G
